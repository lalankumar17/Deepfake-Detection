import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from django.conf import settings

from .gemini_review import GeminiReviewImage, GeminiReviewResult, review_face_crops
from .views import (
    Model,
    device,
    detect_face_locations,
    ensure_bgr_frame,
    ensure_rgb_frame,
    read_static_image,
    train_transforms,
)


FAKE_CLASS_INDEX = 0
REAL_CLASS_INDEX = 1
DEFAULT_LOCAL_FAKE_THRESHOLD = 0.40
DEFAULT_VIDEO_LOCAL_FAKE_THRESHOLD = 0.55
DEFAULT_REVIEW_BAND_LOW = 0.20
DEFAULT_REVIEW_BAND_HIGH = 0.40
DEFAULT_IMAGE_FACE_AREA_RATIO = 0.08
DEFAULT_GEMINI_REAL_OVERRIDE_MIN_CONFIDENCE = 70.0
DEFAULT_GEMINI_REAL_OVERRIDE_MAX_FAKE_PROB = 0.70
DEFAULT_GEMINI_FAKE_OVERRIDE_MIN_CONFIDENCE = 85.0
DEFAULT_GEMINI_FAKE_OVERRIDE_MIN_LOCAL_FAKE_PROB = 0.25
DEFAULT_AI_HINT_MIN_LOCAL_FAKE_PROB = 0.55
DEFAULT_TRACK_MAX_JUMP_MULTIPLIER = 2.6
DEFAULT_TRACK_MIN_JUMP_PIXELS = 42.0

MODEL_BINDINGS = {
    "video": {
        "label": "Video Model",
        "checkpoint": "model_90_acc_20_frames_FF_data.pt",
        "sequence_length": 20,
    },
    "image": {
        "label": "Image Model",
        "checkpoint": "model_90_acc_60_frames_final_data.pt",
        "sequence_length": 60,
    },
}

_MODEL_CACHE = {}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".gif", ".webm", ".avi", ".3gp", ".wmv", ".flv", ".mkv"}


def infer_media_lane_from_path(file_path: str, requested_lane: str) -> str:
    extension = os.path.splitext(file_path or "")[1].lower()
    if extension in IMAGE_EXTENSIONS:
        return "image"
    if extension in VIDEO_EXTENSIONS:
        return "video"
    return requested_lane


@dataclass
class SampleFrame:
    frame_index: int
    frame_bgr: np.ndarray
    frame_rgb: np.ndarray
    face_location: Optional[Tuple[int, int, int, int]]
    face_crop_bgr: np.ndarray
    face_crop_rgb: np.ndarray
    used_face: bool
    heatmap: Optional[np.ndarray] = None
    suspicion_score: float = 0.0


@dataclass
class LocalAnalysisResult:
    label: str
    fake_prob: float
    confidence: float
    face_coverage: float
    sampled_frame_indices: List[int]
    heatmaps: List[np.ndarray]
    review_required: bool
    checkpoint_name: str
    sequence_length: int
    valid_face_crops: int
    samples: List[SampleFrame] = field(default_factory=list)
    review_reason: str = ""
    sampling_summary: str = ""


@dataclass
class FinalDecisionResult:
    label: str
    confidence: float
    decision_source: str
    media_lane: str
    checkpoint_name: str
    local_result: LocalAnalysisResult
    cloud_result: GeminiReviewResult
    output_should_overlay: bool


def get_bool_setting(name: str, default: bool) -> bool:
    value = getattr(settings, name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_float_setting(name: str, default: float) -> float:
    value = getattr(settings, name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_model_binding(media_lane: str) -> dict:
    lane_key = (media_lane or "").strip().lower()
    if lane_key not in MODEL_BINDINGS:
        raise ValueError(f"Unknown media lane '{media_lane}'")
    return MODEL_BINDINGS[lane_key]


def get_bound_checkpoint_path(media_lane: str) -> str:
    binding = get_model_binding(media_lane)
    return os.path.join(settings.PROJECT_DIR, "models", binding["checkpoint"])


def load_bound_model(media_lane: str) -> Tuple[Model, str]:
    checkpoint_path = get_bound_checkpoint_path(media_lane)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Required checkpoint not found: {os.path.basename(checkpoint_path)}"
        )

    cache_key = (
        checkpoint_path,
        device,
        os.path.getmtime(checkpoint_path),
        os.path.getsize(checkpoint_path),
    )
    cached_model = _MODEL_CACHE.get(cache_key)
    if cached_model is not None:
        return cached_model, checkpoint_path

    model = Model(2).cuda() if device == "cuda" else Model(2).cpu()
    map_location = torch.device("cuda") if device == "cuda" else torch.device("cpu")
    model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))
    model.eval()
    _MODEL_CACHE.clear()
    _MODEL_CACHE[cache_key] = model
    return model, checkpoint_path


def face_area(face_location: Tuple[int, int, int, int]) -> int:
    top, right, bottom, left = face_location
    return max(0, bottom - top) * max(0, right - left)


def face_center(face_location: Tuple[int, int, int, int]) -> Tuple[float, float]:
    top, right, bottom, left = face_location
    return ((left + right) / 2.0, (top + bottom) / 2.0)


def select_tracked_face(
    face_locations: List[Tuple[int, int, int, int]],
    previous_face: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    if not face_locations:
        return None

    if previous_face is None:
        return face_locations[0]

    previous_center = face_center(previous_face)
    selected_face = min(
        face_locations,
        key=lambda face: (
            (face_center(face)[0] - previous_center[0]) ** 2
            + (face_center(face)[1] - previous_center[1]) ** 2
        ),
    )
    selected_center = face_center(selected_face)
    center_distance = (
        (selected_center[0] - previous_center[0]) ** 2
        + (selected_center[1] - previous_center[1]) ** 2
    ) ** 0.5
    previous_scale = max(1.0, face_area(previous_face) ** 0.5)
    max_jump = max(
        DEFAULT_TRACK_MIN_JUMP_PIXELS,
        previous_scale * DEFAULT_TRACK_MAX_JUMP_MULTIPLIER,
    )
    if center_distance > max_jump and face_area(selected_face) < face_area(previous_face) * 0.75:
        return None

    return selected_face


def crop_frame(frame_bgr: np.ndarray, face_location: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if face_location is None:
        return frame_bgr.copy()

    top, right, bottom, left = face_location
    height, width = frame_bgr.shape[:2]
    top = max(0, min(int(top), height))
    bottom = max(0, min(int(bottom), height))
    left = max(0, min(int(left), width))
    right = max(0, min(int(right), width))

    if bottom <= top or right <= left:
        return frame_bgr.copy()

    crop = frame_bgr[top:bottom, left:right]
    if crop.size == 0:
        return frame_bgr.copy()
    return crop.copy()


def _read_frame_near_index(
    cap: cv2.VideoCapture,
    frame_index: int,
    total_frames: int,
    search_radius: int = 3,
) -> Optional[Tuple[int, np.ndarray]]:
    candidate_offsets = [0]
    for step in range(1, search_radius + 1):
        candidate_offsets.extend([step, -step])

    attempted_indices = set()
    for offset in candidate_offsets:
        candidate_index = int(frame_index + offset)
        if candidate_index < 0:
            continue
        if total_frames > 0 and candidate_index >= total_frames:
            continue
        if candidate_index in attempted_indices:
            continue

        attempted_indices.add(candidate_index)
        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate_index)
        success, frame = cap.read()
        if success and frame is not None and frame.size > 0:
            return candidate_index, frame

    return None


def sample_video_frames(video_path: str, sequence_length: int) -> Tuple[List[Tuple[int, np.ndarray]], int]:
    cap = cv2.VideoCapture(video_path)
    total_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    sampled_frames: List[Tuple[int, np.ndarray]] = []

    if total_frames > 0:
        frame_indices = np.linspace(0, total_frames - 1, num=sequence_length, dtype=int).tolist()
        unique_indices = sorted(dict.fromkeys(int(index) for index in frame_indices))
        seen_indices = set()
        for frame_index in unique_indices:
            resolved_frame = _read_frame_near_index(cap, frame_index, total_frames)
            if resolved_frame is None:
                continue

            actual_index, frame = resolved_frame
            if actual_index in seen_indices:
                continue

            sampled_frames.append((actual_index, frame))
            seen_indices.add(actual_index)
        cap.release()
        if sampled_frames:
            sampled_frames.sort(key=lambda item: item[0])
            return sampled_frames, total_frames
    else:
        cap.release()

    cap = cv2.VideoCapture(video_path)
    all_frames: List[np.ndarray] = []
    consecutive_failures = 0
    max_consecutive_failures = max(30, sequence_length * 2)
    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame is None or frame.size == 0:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                break
            continue
        consecutive_failures = 0
        all_frames.append(frame)
    cap.release()

    total_frames = len(all_frames)
    if total_frames == 0:
        return [], 0

    frame_indices = np.linspace(0, total_frames - 1, num=sequence_length, dtype=int).tolist()
    unique_indices = sorted(dict.fromkeys(int(index) for index in frame_indices))
    return [(frame_index, all_frames[frame_index]) for frame_index in unique_indices], total_frames


def sample_video_frames_front(video_path: str, sequence_length: int) -> Tuple[List[Tuple[int, np.ndarray]], int]:
    cap = cv2.VideoCapture(video_path)
    total_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    sampled_frames: List[Tuple[int, np.ndarray]] = []
    consecutive_failures = 0
    max_consecutive_failures = max(30, sequence_length * 4)

    while cap.isOpened() and len(sampled_frames) < sequence_length:
        success, frame = cap.read()
        current_index = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        if not success or frame is None or frame.size == 0:
            consecutive_failures += 1
            if total_frames > 0 and current_index + 1 >= total_frames:
                break
            if consecutive_failures >= max_consecutive_failures:
                break
            continue

        consecutive_failures = 0
        sampled_frames.append((current_index, frame))
        if total_frames > 0 and current_index + 1 >= total_frames:
            break

    cap.release()
    return sampled_frames, total_frames


def overlay_heatmap_on_face(face_crop_bgr: np.ndarray, heatmap: Optional[np.ndarray]) -> np.ndarray:
    if heatmap is None or face_crop_bgr.size == 0:
        return face_crop_bgr.copy()

    resized_heatmap = cv2.resize(
        heatmap,
        (face_crop_bgr.shape[1], face_crop_bgr.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    resized_heatmap = cv2.normalize(resized_heatmap, None, 0.0, 1.0, cv2.NORM_MINMAX)
    colored_heatmap = np.zeros((*resized_heatmap.shape, 3), dtype=np.uint8)
    colored_heatmap[..., 2] = 255
    colored_heatmap[..., 1] = np.uint8(np.clip((1.0 - resized_heatmap) * 255, 0, 255))
    alpha = np.clip(resized_heatmap[..., None] * 0.75, 0.0, 0.75)
    blended = (
        face_crop_bgr.astype(np.float32) * (1.0 - alpha)
        + colored_heatmap.astype(np.float32) * alpha
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def build_fake_class_heatmaps(model: Model, feature_maps: torch.Tensor) -> List[np.ndarray]:
    weight_softmax = model.linear1.weight.detach().cpu().numpy()[FAKE_CLASS_INDEX]
    feature_maps_np = feature_maps.detach().cpu().numpy()
    heatmaps = []

    for frame_map in feature_maps_np:
        channels, height, width = frame_map.shape
        cam = np.dot(frame_map.reshape((channels, height * width)).T, weight_softmax.T)
        cam = cam.reshape(height, width).astype(np.float32)
        cam -= float(cam.min())
        max_value = float(cam.max())
        if max_value > 1e-8:
            cam /= max_value
        else:
            cam = np.zeros((height, width), dtype=np.float32)
        cam = cv2.resize(cam, (112, 112), interpolation=cv2.INTER_CUBIC)
        heatmaps.append(cam.astype(np.float32))

    return heatmaps


def _build_video_samples_from_frames(
    sampled_frames: List[Tuple[int, np.ndarray]],
    sequence_length: int,
) -> Tuple[List[SampleFrame], torch.Tensor, int]:
    if not sampled_frames:
        raise ValueError("Could not read frames from the selected video")

    previous_face = None
    previous_crop = None
    valid_face_crops = 0
    sequence_tensors = []
    samples: List[SampleFrame] = []

    for frame_index, raw_frame in sampled_frames:
        frame_bgr = ensure_bgr_frame(raw_frame)
        if frame_bgr is None:
            continue

        frame_rgb = ensure_rgb_frame(frame_bgr)
        face_locations = detect_face_locations(frame_bgr)
        selected_face = select_tracked_face(face_locations, previous_face)
        used_face = selected_face is not None

        if selected_face is not None:
            previous_face = selected_face
            previous_crop = crop_frame(frame_bgr, selected_face)
            model_crop = previous_crop.copy()
            display_crop = previous_crop.copy()
            valid_face_crops += 1
        elif previous_crop is not None:
            model_crop = previous_crop.copy()
            display_crop = None
        else:
            model_crop = frame_bgr.copy()
            display_crop = None

        if display_crop is not None:
            display_crop_bgr = display_crop.copy()
            display_crop_rgb = ensure_rgb_frame(display_crop)
        else:
            display_crop_bgr = np.empty((0, 0, 3), dtype=np.uint8)
            display_crop_rgb = np.empty((0, 0, 3), dtype=np.uint8)

        sequence_tensors.append(train_transforms(model_crop))
        samples.append(
            SampleFrame(
                frame_index=frame_index,
                frame_bgr=frame_bgr.copy(),
                frame_rgb=frame_rgb.copy(),
                face_location=selected_face,
                face_crop_bgr=display_crop_bgr,
                face_crop_rgb=display_crop_rgb,
                used_face=used_face,
            )
        )

    if not sequence_tensors:
        raise ValueError("Could not extract a valid sampled sequence from the selected video")

    last_tensor = sequence_tensors[-1]
    while len(sequence_tensors) < sequence_length:
        sequence_tensors.append(last_tensor.clone())

    sequence_tensor = torch.stack(sequence_tensors[:sequence_length]).unsqueeze(0)
    return samples, sequence_tensor, valid_face_crops


def _build_video_samples(
    video_path: str,
    sequence_length: int,
    sampling_mode: str = "uniform",
) -> Tuple[List[SampleFrame], torch.Tensor, int, int]:
    if sampling_mode == "legacy-first":
        sampled_frames, total_frames = sample_video_frames_front(video_path, sequence_length)
    else:
        sampled_frames, total_frames = sample_video_frames(video_path, sequence_length)

    samples, sequence_tensor, valid_face_crops = _build_video_samples_from_frames(
        sampled_frames,
        sequence_length,
    )
    return samples, sequence_tensor, total_frames, valid_face_crops


def _build_image_samples(image_path: str, sequence_length: int) -> Tuple[List[SampleFrame], torch.Tensor, int, int, float]:
    frame_bgr = read_static_image(image_path)
    frame_bgr = ensure_bgr_frame(frame_bgr)
    if frame_bgr is None:
        raise ValueError("Could not read the selected image")

    frame_rgb = ensure_rgb_frame(frame_bgr)
    face_locations = detect_face_locations(frame_bgr)
    selected_face = select_tracked_face(face_locations)
    model_crop = crop_frame(frame_bgr, selected_face)
    valid_face_crops = 1 if selected_face is not None else 0

    frame_height, frame_width = frame_bgr.shape[:2]
    image_area = max(1, frame_height * frame_width)
    face_area_ratio = 0.0
    if selected_face is not None:
        face_area_ratio = face_area(selected_face) / float(image_area)

    base_tensor = train_transforms(model_crop)
    sequence_tensor = torch.stack([base_tensor.clone() for _ in range(sequence_length)]).unsqueeze(0)
    sample = SampleFrame(
        frame_index=0,
        frame_bgr=frame_bgr.copy(),
        frame_rgb=frame_rgb.copy(),
        face_location=selected_face,
        face_crop_bgr=model_crop.copy(),
        face_crop_rgb=ensure_rgb_frame(model_crop),
        used_face=selected_face is not None,
    )
    return [sample], sequence_tensor, 1, valid_face_crops, face_area_ratio


def _assign_heatmaps(samples: List[SampleFrame], heatmaps: List[np.ndarray]) -> None:
    for index, sample in enumerate(samples):
        if index >= len(heatmaps):
            break
        sample.heatmap = heatmaps[index]
        sample.suspicion_score = float(np.mean(heatmaps[index])) if heatmaps[index] is not None else 0.0


def _build_local_result(
    model: Model,
    sequence_tensor: torch.Tensor,
    samples: List[SampleFrame],
    checkpoint_path: str,
    sequence_length: int,
    valid_face_crops: int,
    face_coverage: float,
    local_fake_threshold: float,
    review_required: bool,
    sampling_summary: str = "",
) -> LocalAnalysisResult:
    inference_tensor = sequence_tensor.cuda() if device == "cuda" else sequence_tensor.cpu()
    with torch.no_grad():
        feature_maps, logits = model(inference_tensor)
        probabilities = torch.softmax(logits, dim=1)

    fake_prob = float(probabilities[0, FAKE_CLASS_INDEX].item())
    real_prob = float(probabilities[0, REAL_CLASS_INDEX].item())
    local_label = "FAKE" if fake_prob >= local_fake_threshold else "REAL"
    local_confidence = fake_prob * 100.0 if local_label == "FAKE" else real_prob * 100.0
    heatmaps = build_fake_class_heatmaps(model, feature_maps)
    _assign_heatmaps(samples, heatmaps)

    return LocalAnalysisResult(
        label=local_label,
        fake_prob=fake_prob,
        confidence=round(local_confidence, 1),
        face_coverage=round(face_coverage, 3),
        sampled_frame_indices=[sample.frame_index for sample in samples],
        heatmaps=heatmaps,
        review_required=review_required,
        checkpoint_name=os.path.basename(checkpoint_path),
        sequence_length=sequence_length,
        valid_face_crops=valid_face_crops,
        samples=samples,
        sampling_summary=sampling_summary,
    )


def _reason_mentions_generator_watermark(reason: str) -> bool:
    normalized_reason = (reason or "").strip().lower()
    if not normalized_reason:
        return False

    watermark_keywords = (
        "watermark",
        "gemini",
        "sparkle",
        "star mark",
        "generator",
        "ai video badge",
        "generator mark",
        "generator logo",
        "generator branding",
        "sora",
        "veo",
        "runway",
        "pika",
        "kling",
        "synthid",
    )
    return any(keyword in normalized_reason for keyword in watermark_keywords)


def _crop_region_by_ratio(
    frame_bgr: np.ndarray,
    left_ratio: float,
    top_ratio: float,
    right_ratio: float,
    bottom_ratio: float,
) -> np.ndarray:
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr

    height, width = frame_bgr.shape[:2]
    left = max(0, min(width - 1, int(width * left_ratio)))
    right = max(left + 1, min(width, int(width * right_ratio)))
    top = max(0, min(height - 1, int(height * top_ratio)))
    bottom = max(top + 1, min(height, int(height * bottom_ratio)))
    return frame_bgr[top:bottom, left:right].copy()


def _detect_generator_sparkle_signature(frame_bgr: np.ndarray) -> bool:
    if frame_bgr is None or frame_bgr.size == 0:
        return False

    frame_height, frame_width = frame_bgr.shape[:2]
    if frame_height < 120 or frame_width < 120:
        return False

    left = int(frame_width * 0.68)
    top = int(frame_height * 0.68)
    corner = frame_bgr[top:frame_height, left:frame_width]
    if corner.size == 0:
        return False

    crop_height, crop_width = corner.shape[:2]
    hsv = cv2.cvtColor(corner, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    min_frame_dimension = min(frame_height, frame_width)
    min_box_size = max(24.0, min_frame_dimension * 0.018)
    max_box_size = min_frame_dimension * 0.145
    candidates = []

    for gray_threshold in (150, 160, 170, 180):
        bright_low_saturation = ((gray > gray_threshold) & (hsv[:, :, 1] < 120)).astype("uint8") * 255
        bright_low_saturation = cv2.morphologyEx(
            bright_low_saturation,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        contours, _ = cv2.findContours(
            bright_low_saturation,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100.0:
                continue

            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_width <= 0 or box_height <= 0:
                continue

            center_x = (x + box_width / 2.0) / float(crop_width)
            center_y = (y + box_height / 2.0) / float(crop_height)
            aspect_ratio = max(
                box_width / float(max(1, box_height)),
                box_height / float(max(1, box_width)),
            )
            extent = area / float(max(1, box_width * box_height))

            if center_x < 0.72 or center_y < 0.48:
                continue
            if not (min_box_size <= box_width <= max_box_size):
                continue
            if not (min_box_size <= box_height <= max_box_size):
                continue
            if aspect_ratio > 1.85:
                continue
            if not (0.16 <= extent <= 0.74):
                continue

            global_center_x = (left + x + box_width / 2.0) / float(frame_width)
            global_center_y = (top + y + box_height / 2.0) / float(frame_height)
            candidates.append((global_center_x, global_center_y, gray_threshold))

    for center_x, center_y, _threshold in candidates:
        matching_thresholds = {
            other_threshold
            for other_x, other_y, other_threshold in candidates
            if abs(other_x - center_x) <= 0.035 and abs(other_y - center_y) <= 0.035
        }
        if len(matching_thresholds) >= 2:
            return True

    return False


def _detect_ai_generated_image_hint(
    file_path: str,
    media_lane: str,
    local_result: LocalAnalysisResult,
) -> Tuple[bool, float, str]:
    normalized_name = os.path.basename(file_path or "").lower()
    normalized_name = normalized_name.replace("-", "_").replace(" ", "_")
    filename_keywords = (
        "gemini_generated",
        "ai_generated",
        "generated_image",
        "generated_photo",
        "dall_e",
        "dalle",
        "midjourney",
        "stable_diffusion",
        "comfyui",
        "leonardo_ai",
        "ideogram",
    )
    for keyword in filename_keywords:
        if keyword in normalized_name:
            return True, 99.0, f"filename indicates AI-generated media ({keyword})"

    if not local_result.samples:
        return False, 0.0, ""

    candidate_positions = np.linspace(
        0,
        len(local_result.samples) - 1,
        num=min(3, len(local_result.samples)),
        dtype=int,
    ).tolist()

    min_local_fake_prob = get_float_setting(
        "AI_HINT_MIN_LOCAL_FAKE_PROB",
        DEFAULT_AI_HINT_MIN_LOCAL_FAKE_PROB,
    )
    allow_unconditional_generator_mark = media_lane == "image"

    for position in candidate_positions:
        frame = local_result.samples[position].frame_bgr
        if frame is None or frame.size == 0:
            continue

        if (
            _detect_generator_sparkle_signature(frame)
            and (allow_unconditional_generator_mark or local_result.fake_prob >= min_local_fake_prob)
        ):
            return True, 99.0, "visible AI generator sparkle/star watermark signature"

    if local_result.fake_prob < min_local_fake_prob:
        return False, 0.0, ""

    for position in candidate_positions:
        frame = local_result.samples[position].frame_bgr
        if frame is None or frame.size == 0:
            continue

        # Conservative visual hint for generated-image/video signatures in the bottom-right corner.
        height, width = frame.shape[:2]
        corner = frame[int(height * 0.72):height, int(width * 0.72):width]
        if corner.size == 0:
            continue

        hsv = cv2.cvtColor(corner, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
        bright_low_saturation = ((gray > 225) & (hsv[:, :, 1] < 80)).astype("uint8") * 255
        contours, _ = cv2.findContours(bright_low_saturation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        crop_height, crop_width = corner.shape[:2]
        plausible_marks = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if not (6.0 <= area <= 1800.0):
                continue
            if box_width < 3 or box_height < 5:
                continue
            center_x = x + box_width / 2.0
            center_y = y + box_height / 2.0
            if center_x >= crop_width * 0.35 and center_y >= crop_height * 0.10:
                plausible_marks += 1

        if plausible_marks >= 2:
            return True, 95.0, "bottom-right bright low-saturation generator watermark/signature candidate"

    return False, 0.0, ""


def _select_review_images(local_result: LocalAnalysisResult, media_lane: str) -> List[GeminiReviewImage]:
    samples = [
        sample
        for sample in local_result.samples
        if sample.face_crop_bgr is not None and sample.face_crop_bgr.size > 0
    ]
    if not samples:
        return []

    review_images: List[GeminiReviewImage] = []
    used_labels = set()

    def add_review_image(label: str, image_bgr: np.ndarray) -> None:
        if label in used_labels:
            return
        review_images.append(GeminiReviewImage(label=label, image_bgr=image_bgr))
        used_labels.add(label)

    if media_lane == "image":
        sample = samples[0]
        add_review_image(
            "Full image context - inspect whole scene, hands, text, corners, and generator watermark",
            sample.frame_bgr,
        )
        add_review_image(
            "Bottom-right corner watermark/signature check",
            _crop_region_by_ratio(sample.frame_bgr, 0.60, 0.60, 1.0, 1.0),
        )
        add_review_image(
            "Lower scene object/text check",
            _crop_region_by_ratio(sample.frame_bgr, 0.20, 0.55, 0.95, 1.0),
        )
        add_review_image("Main face crop - raw face", sample.face_crop_bgr)
        add_review_image(
            "Main face crop - local fake heatmap overlay",
            overlay_heatmap_on_face(sample.face_crop_bgr, sample.heatmap),
        )
        return review_images

    # Prefer real detected face crops so Gemini reviews the full tracked sequence,
    # not repeated fallback crops copied into missing-face slots.
    used_face_samples = [sample for sample in samples if sample.used_face]
    selected_samples = used_face_samples or samples

    selected_positions = np.linspace(
        0,
        len(selected_samples) - 1,
        num=min(4, len(selected_samples)),
        dtype=int,
    ).tolist()
    for review_index, sample_position in enumerate(selected_positions, start=1):
        sample = selected_samples[sample_position]
        add_review_image(
            f"Crop {review_index} - frame {sample.frame_index} - suspicion {sample.suspicion_score:.3f}",
            sample.face_crop_bgr,
        )

    context_positions = np.linspace(0, len(samples) - 1, num=min(2, len(samples)), dtype=int).tolist()
    for position in context_positions:
        sample = samples[position]
        add_review_image(
            f"Context frame {sample.frame_index} - watermark check",
            sample.frame_bgr,
        )

    return review_images


def analyze_media(file_path: str, media_lane: str, review_required_override: Optional[bool] = None) -> FinalDecisionResult:
    media_lane = infer_media_lane_from_path(file_path, media_lane)
    binding = get_model_binding(media_lane)
    sequence_length = int(binding["sequence_length"])
    model, checkpoint_path = load_bound_model(media_lane)
    if media_lane == "video":
        local_fake_threshold = get_float_setting(
            "VIDEO_LOCAL_FAKE_THRESHOLD",
            DEFAULT_VIDEO_LOCAL_FAKE_THRESHOLD,
        )
    else:
        local_fake_threshold = get_float_setting("LOCAL_FAKE_THRESHOLD", DEFAULT_LOCAL_FAKE_THRESHOLD)
    review_band_low = get_float_setting("GEMINI_REVIEW_BAND_LOW", DEFAULT_REVIEW_BAND_LOW)
    review_band_high = get_float_setting("GEMINI_REVIEW_BAND_HIGH", DEFAULT_REVIEW_BAND_HIGH)
    real_override_min_confidence = get_float_setting(
        "GEMINI_REAL_OVERRIDE_MIN_CONFIDENCE",
        DEFAULT_GEMINI_REAL_OVERRIDE_MIN_CONFIDENCE,
    )
    real_override_max_fake_prob = get_float_setting(
        "GEMINI_REAL_OVERRIDE_MAX_FAKE_PROB",
        DEFAULT_GEMINI_REAL_OVERRIDE_MAX_FAKE_PROB,
    )
    fake_override_min_confidence = get_float_setting(
        "GEMINI_FAKE_OVERRIDE_MIN_CONFIDENCE",
        DEFAULT_GEMINI_FAKE_OVERRIDE_MIN_CONFIDENCE,
    )
    fake_override_min_local_fake_prob = get_float_setting(
        "GEMINI_FAKE_OVERRIDE_MIN_LOCAL_FAKE_PROB",
        DEFAULT_GEMINI_FAKE_OVERRIDE_MIN_LOCAL_FAKE_PROB,
    )
    review_required = (
        bool(review_required_override)
        if review_required_override is not None
        else get_bool_setting("ENABLE_GEMINI_REVIEW", True)
    )
    dual_video_sampling = get_bool_setting("ENABLE_DUAL_VIDEO_SAMPLING", False)

    if media_lane == "video":
        candidate_results: List[Tuple[str, LocalAnalysisResult]] = []
        sampling_errors: List[str] = []

        sampling_strategies = [("uniform", "uniform")]
        if dual_video_sampling:
            sampling_strategies.insert(0, ("legacy-first", "legacy-first"))

        for sampling_mode, sampling_label in sampling_strategies:
            try:
                strategy_samples, strategy_tensor, _, strategy_valid_face_crops = _build_video_samples(
                    file_path,
                    sequence_length,
                    sampling_mode=sampling_mode,
                )
                strategy_face_coverage = strategy_valid_face_crops / float(sequence_length)
                strategy_result = _build_local_result(
                    model,
                    strategy_tensor,
                    strategy_samples,
                    checkpoint_path,
                    sequence_length,
                    strategy_valid_face_crops,
                    strategy_face_coverage,
                    local_fake_threshold,
                    review_required,
                    sampling_summary=sampling_label,
                )
                candidate_results.append((sampling_label, strategy_result))
            except ValueError as exc:
                sampling_errors.append(f"{sampling_label} unavailable ({exc})")

        if not candidate_results:
            raise ValueError("Could not build any valid video sample clip from the selected file")

        selected_label, local_result = max(
            candidate_results,
            key=lambda item: (item[1].fake_prob, item[1].face_coverage),
        )
        samples = local_result.samples
        valid_face_crops = local_result.valid_face_crops
        face_coverage = local_result.face_coverage
        image_face_ratio = 0.0
        score_parts = [f"{label}={result.fake_prob:.3f}" for label, result in candidate_results]
        score_parts.append(f"selected={selected_label}")
        if sampling_errors:
            score_parts.extend(sampling_errors)
        local_result.sampling_summary = ", ".join(score_parts)
    else:
        samples, sequence_tensor, _, valid_face_crops, image_face_ratio = _build_image_samples(
            file_path, sequence_length
        )
        face_coverage = 1.0 if valid_face_crops > 0 else 0.0
        local_result = _build_local_result(
            model,
            sequence_tensor,
            samples,
            checkpoint_path,
            sequence_length,
            valid_face_crops,
            face_coverage,
            local_fake_threshold,
            review_required,
        )

    review_triggers = [
        "Gemini review enabled" if review_required else "Gemini review not selected",
        f"local fake probability {local_result.fake_prob:.3f}",
    ]
    if media_lane == "video" and local_result.sampling_summary:
        review_triggers.append(f"local sampling {local_result.sampling_summary}")
    if review_band_low <= local_result.fake_prob < review_band_high:
        review_triggers.append("local fake probability in Gemini review band")
    if face_coverage < 0.70:
        review_triggers.append(f"face coverage low ({face_coverage:.2f})")
    minimum_video_face_crops = max(12, int(np.ceil(sequence_length * 0.4)))
    if media_lane == "video" and valid_face_crops < minimum_video_face_crops:
        review_triggers.append(f"valid chosen face crops low ({valid_face_crops}/{sequence_length})")
    if media_lane == "image" and (valid_face_crops == 0 or image_face_ratio < DEFAULT_IMAGE_FACE_AREA_RATIO):
        review_triggers.append("image face crop missing or too small")

    local_result.review_reason = "; ".join(review_triggers)

    cloud_result = GeminiReviewResult(status="skipped", reason="Gemini review disabled")
    if review_required:
        review_images = _select_review_images(local_result, media_lane)
        cloud_result = review_face_crops(media_lane, review_images, local_result.fake_prob)

    ai_hint_detected, ai_hint_confidence, ai_hint_reason = _detect_ai_generated_image_hint(
        file_path,
        media_lane,
        local_result,
    )
    if ai_hint_detected:
        local_result.review_reason = (
            local_result.review_reason + "; " + ai_hint_reason
            if local_result.review_reason
            else ai_hint_reason
        )

    final_label = local_result.label
    final_confidence = local_result.confidence
    decision_source = "Local + Gemini" if review_required else "Local Only"

    if ai_hint_detected:
        final_label = "FAKE"
        final_confidence = round(ai_hint_confidence, 1)
        decision_source = "Local + Gemini" if review_required else "Local Only"
    elif cloud_result.status == "reviewed":
        decision_source = "Local + Gemini"
        watermark_override = _reason_mentions_generator_watermark(cloud_result.reason)
        fake_override_allowed = (
            watermark_override
            or (
                cloud_result.confidence >= fake_override_min_confidence
                and local_result.fake_prob >= fake_override_min_local_fake_prob
            )
        )
        if cloud_result.label == "FAKE" and fake_override_allowed:
            final_label = "FAKE"
            final_confidence = round(max(cloud_result.confidence, 95.0 if watermark_override else 0.0), 1)
        elif (
            cloud_result.label == "REAL"
            and cloud_result.confidence >= real_override_min_confidence
            and local_result.fake_prob <= real_override_max_fake_prob
        ):
            final_label = "REAL"
            final_confidence = round(cloud_result.confidence, 1)
        else:
            final_label = local_result.label
            final_confidence = local_result.confidence
    elif review_required:
        decision_source = "Local + Gemini"

    return FinalDecisionResult(
        label=final_label,
        confidence=round(final_confidence, 1),
        decision_source=decision_source,
        media_lane=media_lane,
        checkpoint_name=os.path.basename(checkpoint_path),
        local_result=local_result,
        cloud_result=cloud_result,
        output_should_overlay=final_label == "FAKE" and bool(local_result.heatmaps),
    )
