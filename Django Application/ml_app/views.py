from django.shortcuts import render, redirect
import torch
from torchvision import transforms, models
import os
import numpy as np
import cv2
try:
    import face_recognition
except ImportError:
    face_recognition = None
import time
from torch import nn
import shutil
from PIL import Image as pImage
from django.conf import settings
from .forms import VideoUploadForm

index_template_name = 'index.html'
predict_template_name = 'predict.html'
about_template_name = "about.html"

im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])

FACE_DETECTION_BACKEND = "face_recognition" if face_recognition is not None else "opencv"
FACE_DETECTION_FALLBACK_REASON = None

def ensure_uint8_frame(frame):
    if frame is None:
        return None

    frame = np.asarray(frame)
    if frame.size == 0:
        return None

    if frame.dtype != np.uint8:
        frame = np.nan_to_num(frame)
        max_value = float(frame.max())
        min_value = float(frame.min())
        if min_value >= 0.0 and max_value <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return frame

def ensure_bgr_frame(frame):
    frame = ensure_uint8_frame(frame)
    if frame is None:
        return None

    if frame.ndim == 2:
        return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

    if frame.ndim == 3:
        channels = frame.shape[2]
        if channels == 1:
            return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        if channels == 3:
            return np.ascontiguousarray(frame)
        if channels == 4:
            return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR))

    raise ValueError("Unsupported image type after normalization")

def ensure_rgb_frame(frame):
    bgr_frame = ensure_bgr_frame(frame)
    if bgr_frame is None:
        return None
    return np.ascontiguousarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))

def read_static_image(path):
    try:
        with pImage.open(path) as image:
            rgb_image = image.convert('RGB')
            return np.ascontiguousarray(cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR))
    except Exception:
        return ensure_bgr_frame(cv2.imread(path, cv2.IMREAD_UNCHANGED))

def get_face_cascades():
    if hasattr(get_face_cascades, "_cascades"):
        return get_face_cascades._cascades

    cascade_files = [
        'haarcascade_frontalface_alt.xml',
        'haarcascade_frontalface_alt2.xml',
        'haarcascade_frontalface_default.xml',
        'haarcascade_profileface.xml',
    ]
    cascades = []
    missing_paths = []
    for cascade_file in cascade_files:
        cascade_path = os.path.join(cv2.data.haarcascades, cascade_file)
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            missing_paths.append(cascade_path)
            continue
        cascades.append((cascade_file, cascade))

    if not cascades:
        raise RuntimeError(f"Could not load Haar cascades from {missing_paths}")

    get_face_cascades._cascades = cascades
    return cascades

def get_face_cascade():
    return get_face_cascades()[0][1]

def get_eye_cascades():
    if hasattr(get_eye_cascades, "_cascades"):
        return get_eye_cascades._cascades

    cascade_files = [
        'haarcascade_eye.xml',
        'haarcascade_eye_tree_eyeglasses.xml',
    ]
    cascades = []
    for cascade_file in cascade_files:
        cascade_path = os.path.join(cv2.data.haarcascades, cascade_file)
        cascade = cv2.CascadeClassifier(cascade_path)
        if not cascade.empty():
            cascades.append((cascade_file, cascade))

    get_eye_cascades._cascades = cascades
    return cascades

def get_detection_int_setting(name, default):
    value = os.getenv(name, getattr(settings, name, default))
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def get_detection_float_setting(name, default):
    value = os.getenv(name, getattr(settings, name, default))
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def normalize_face_location(face_location, frame_shape):
    if face_location is None or frame_shape is None:
        return None

    frame_height, frame_width = frame_shape[:2]
    if frame_height <= 0 or frame_width <= 0:
        return None

    top, right, bottom, left = face_location
    top = max(0, min(int(top), frame_height - 1))
    bottom = max(0, min(int(bottom), frame_height))
    left = max(0, min(int(left), frame_width - 1))
    right = max(0, min(int(right), frame_width))
    if bottom <= top or right <= left:
        return None
    return top, right, bottom, left

def face_location_area_ratio(face_location, frame_shape):
    normalized = normalize_face_location(face_location, frame_shape)
    if normalized is None:
        return 0.0

    frame_height, frame_width = frame_shape[:2]
    top, right, bottom, left = normalized
    return ((bottom - top) * (right - left)) / float(max(1, frame_height * frame_width))

def face_location_main_score(face_location, frame_shape):
    normalized = normalize_face_location(face_location, frame_shape)
    if normalized is None:
        return 0.0

    frame_height, frame_width = frame_shape[:2]
    top, right, bottom, left = normalized
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    frame_center_x = frame_width / 2.0
    frame_center_y = frame_height / 2.0
    max_distance = max(1.0, (frame_center_x ** 2 + frame_center_y ** 2) ** 0.5)
    center_distance = ((center_x - frame_center_x) ** 2 + (center_y - frame_center_y) ** 2) ** 0.5
    center_score = max(0.0, 1.0 - (center_distance / max_distance))
    area_ratio = face_location_area_ratio(normalized, frame_shape)

    # Favor the main subject face over map/text/logo false positives in corners.
    return area_ratio * (0.35 + 0.65 * center_score)

def face_location_eye_score(face_location, gray_frame):
    normalized = normalize_face_location(face_location, gray_frame.shape if gray_frame is not None else None)
    if normalized is None or gray_frame is None or gray_frame.size == 0:
        return 0.0

    top, right, bottom, left = normalized
    face_crop = gray_frame[top:bottom, left:right]
    if face_crop.size == 0:
        return 0.0

    face_height, face_width = face_crop.shape[:2]
    if face_height < 36 or face_width < 36:
        return 0.0

    upper_face = face_crop[:max(1, int(face_height * 0.72)), :]
    equalized_face = cv2.equalizeHist(upper_face)
    min_eye_size = max(8, int(min(face_height, face_width) * 0.12))
    best_score = 0.0

    for _cascade_name, cascade in get_eye_cascades():
        try:
            detections = cascade.detectMultiScale(
                equalized_face,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(min_eye_size, min_eye_size),
            )
        except cv2.error:
            continue

        score = 0.0
        for x, y, width, height in detections:
            center_x = x + width / 2.0
            center_y = y + height / 2.0
            if not (face_width * 0.10 <= center_x <= face_width * 0.90):
                continue
            if not (face_height * 0.06 <= center_y <= face_height * 0.72):
                continue
            area_ratio = (width * height) / float(max(1, face_width * face_height))
            score += 1.0 + min(1.0, area_ratio * 10.0)

        best_score = max(best_score, score)

    return best_score

def is_plausible_face_location(face_location, frame_shape):
    normalized = normalize_face_location(face_location, frame_shape)
    if normalized is None:
        return False

    frame_height, frame_width = frame_shape[:2]
    top, right, bottom, left = normalized
    face_width = right - left
    face_height = bottom - top
    min_side = max(20, int(min(frame_height, frame_width) * 0.035))
    min_area_ratio = get_detection_float_setting("FACE_BOX_MIN_AREA_RATIO", 0.004)
    max_area_ratio = get_detection_float_setting("FACE_BOX_MAX_AREA_RATIO", 0.65)
    aspect_ratio = face_height / float(max(1, face_width))
    area_ratio = face_location_area_ratio(normalized, frame_shape)

    return (
        face_width >= min_side
        and face_height >= min_side
        and min_area_ratio <= area_ratio <= max_area_ratio
        and 0.55 <= aspect_ratio <= 1.95
    )

def _gray_frame_for_face_scoring(frame):
    if frame is None:
        return None
    frame = ensure_uint8_frame(frame)
    if frame is None:
        return None
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
    return None

def filter_face_locations(face_locations, frame_shape, limit=4, frame=None):
    scored_locations = []
    seen = set()
    gray_frame = _gray_frame_for_face_scoring(frame)
    for face_location in face_locations or []:
        normalized = normalize_face_location(face_location, frame_shape)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        if is_plausible_face_location(normalized, frame_shape):
            eye_score = face_location_eye_score(normalized, gray_frame)
            scored_locations.append((normalized, eye_score))

    has_eye_validated_face = any(eye_score > 0.0 for _loc, eye_score in scored_locations)
    if has_eye_validated_face:
        scored_locations.sort(
            key=lambda item: (
                item[1] > 0.0,
                item[1],
                face_location_main_score(item[0], frame_shape),
            ),
            reverse=True,
        )
    else:
        scored_locations.sort(key=lambda item: face_location_main_score(item[0], frame_shape), reverse=True)
    return [location for location, _eye_score in scored_locations[:limit]]

def refine_face_location_for_image(frame, face_location):
    if frame is None or face_location is None:
        return face_location

    frame_bgr = ensure_bgr_frame(frame)
    normalized = normalize_face_location(face_location, frame_bgr.shape if frame_bgr is not None else None)
    if frame_bgr is None or normalized is None:
        return normalized

    face_locations = detect_face_locations(frame_bgr)
    if not face_locations:
        return normalized

    if normalized in face_locations:
        return normalized

    return face_locations[0]

def cascade_detect(cascade, gray_frame, min_size, mirrored=False):
    min_neighbors = get_detection_int_setting("FACE_CASCADE_MIN_NEIGHBORS", 3)
    try:
        detected_faces, _reject_levels, weights = cascade.detectMultiScale3(
            gray_frame,
            scaleFactor=1.05,
            minNeighbors=min_neighbors,
            minSize=(min_size, min_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
            outputRejectLevels=True,
        )
    except (AttributeError, cv2.error):
        detected_faces = cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.05,
            minNeighbors=min_neighbors,
            minSize=(min_size, min_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        weights = [1.0] * len(detected_faces)

    frame_width = gray_frame.shape[1]
    detections = []
    for (x, y, w, h), weight in zip(detected_faces, weights):
        if mirrored:
            x = frame_width - x - w
        detections.append((float(weight), int(x), int(y), int(w), int(h)))
    detections.sort(reverse=True)
    return detections

def detections_to_locations(detections):
    return [
        (int(y), int(x + w), int(y + h), int(x))
        for _weight, x, y, w, h in detections
    ]

def detect_face_locations(frame):
    global FACE_DETECTION_BACKEND
    global FACE_DETECTION_FALLBACK_REASON

    rgb_frame = ensure_rgb_frame(frame)
    if rgb_frame is None:
        return []

    if FACE_DETECTION_BACKEND == "face_recognition" and face_recognition is not None:
        try:
            face_locations = face_recognition.face_locations(np.ascontiguousarray(rgb_frame))
            filtered_locations = filter_face_locations(face_locations, rgb_frame.shape, frame=rgb_frame)
            if filtered_locations:
                return filtered_locations
        except RuntimeError as exc:
            FACE_DETECTION_BACKEND = "opencv"
            FACE_DETECTION_FALLBACK_REASON = str(exc)
        except Exception:
            FACE_DETECTION_BACKEND = "opencv"
            FACE_DETECTION_FALLBACK_REASON = "face_recognition backend failed unexpectedly"

    gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    equalized_frame = cv2.equalizeHist(gray_frame)
    frame_height, frame_width = gray_frame.shape[:2]
    min_size = max(24, int(min(frame_height, frame_width) * 0.05))
    all_detections = []

    for variant in (gray_frame, equalized_frame):
        for cascade_name, cascade in get_face_cascades():
            detections = cascade_detect(cascade, variant, min_size)
            if detections:
                all_detections.extend(detections)

            if cascade_name == 'haarcascade_profileface.xml':
                mirrored_variant = cv2.flip(variant, 1)
                detections = cascade_detect(cascade, mirrored_variant, min_size, mirrored=True)
                if detections:
                    all_detections.extend(detections)

    if all_detections:
        all_detections.sort(reverse=True)
        return filter_face_locations(detections_to_locations(all_detections[:16]), gray_frame.shape, frame=gray_frame)

    return []

def resize_frame_for_preview(frame, max_dimension=960):
    frame = ensure_bgr_frame(frame)
    if frame is None:
        return None

    height, width = frame.shape[:2]
    largest_dimension = max(height, width)
    if largest_dimension <= max_dimension:
        return frame

    scale = max_dimension / float(largest_dimension)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

def scale_face_location(face_location, scale_x=1.0, scale_y=1.0):
    if face_location is None:
        return None
    top, right, bottom, left = face_location
    return (
        int(round(top * scale_y)),
        int(round(right * scale_x)),
        int(round(bottom * scale_y)),
        int(round(left * scale_x)),
    )

def expand_face_location_for_overlay(face_location, frame_shape):
    normalized = normalize_face_location(face_location, frame_shape)
    if normalized is None:
        return None

    frame_height, frame_width = frame_shape[:2]
    top, right, bottom, left = normalized
    face_width = right - left
    face_height = bottom - top

    horizontal_padding = int(round(face_width * 0.12))
    top_padding = int(round(face_height * 0.08))
    bottom_padding = int(round(face_height * 0.10))

    expanded_top = max(0, top - top_padding)
    expanded_bottom = min(frame_height, bottom + bottom_padding)
    expanded_left = max(0, left - horizontal_padding)
    expanded_right = min(frame_width, right + horizontal_padding)

    return normalize_face_location(
        (expanded_top, expanded_right, expanded_bottom, expanded_left),
        frame_shape,
    )

def draw_face_read_box(frame, face_location, output_label, confidence):
    if frame is None or face_location is None:
        return frame

    frame_height, frame_width = frame.shape[:2]
    overlay_location = expand_face_location_for_overlay(face_location, frame.shape)
    if overlay_location is None:
        return frame

    top, right, bottom, left = overlay_location
    top = max(0, min(int(top), frame_height - 1))
    bottom = max(0, min(int(bottom), frame_height - 1))
    left = max(0, min(int(left), frame_width - 1))
    right = max(0, min(int(right), frame_width - 1))
    if bottom <= top or right <= left:
        return frame

    color = (255, 255, 0) if output_label == "REAL" else (0, 0, 255)
    thickness = max(3, int(round(max(frame_width, frame_height) / 360)))
    label_text = f"READ: {output_label} {confidence}%"
    cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)

    text_scale = max(0.58, min(0.85, frame_width / 900.0))
    text_thickness = max(2, thickness - 1)
    (text_width, text_height), baseline = cv2.getTextSize(
        label_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        text_thickness,
    )
    text_y = max(text_height + 12, top - 10)
    text_x = max(0, min(left, frame_width - text_width - 8))
    cv2.rectangle(
        frame,
        (text_x, text_y - text_height - baseline - 6),
        (text_x + text_width + 8, text_y + baseline + 4),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        frame,
        label_text,
        (text_x + 4, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        color,
        text_thickness,
    )
    return frame

def apply_tamper_overlay(frame, face_location, heatmap, output_label):
    if output_label != "FAKE" or heatmap is None or frame is None or face_location is None:
        return False

    frame_height, frame_width = frame.shape[:2]
    overlay_location = expand_face_location_for_overlay(face_location, frame.shape)
    if overlay_location is None:
        return False

    top, right, bottom, left = overlay_location
    top = max(0, min(int(top), frame_height))
    bottom = max(0, min(int(bottom), frame_height))
    left = max(0, min(int(left), frame_width))
    right = max(0, min(int(right), frame_width))
    if bottom <= top or right <= left:
        return False

    roi = frame[top:bottom, left:right]
    if roi.size == 0:
        return False

    resized_heatmap = cv2.resize(heatmap, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_CUBIC)
    resized_heatmap = cv2.GaussianBlur(resized_heatmap, (0, 0), sigmaX=1.5, sigmaY=1.5)
    resized_heatmap = cv2.normalize(resized_heatmap, None, 0.0, 1.0, cv2.NORM_MINMAX)

    threshold = max(0.25, float(np.quantile(resized_heatmap, 0.50)))
    tamper_mask = np.clip((resized_heatmap - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
    if float(tamper_mask.max()) <= 0.02:
        return False

    face_mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.float32)
    center = (roi.shape[1] // 2, roi.shape[0] // 2)
    axes = (max(1, int(roi.shape[1] * 0.38)), max(1, int(roi.shape[0] * 0.48)))
    cv2.ellipse(face_mask, center, axes, 0, 0, 360, 1.0, -1)
    face_mask = cv2.GaussianBlur(
        face_mask,
        (0, 0),
        sigmaX=max(1.0, roi.shape[1] / 18.0),
        sigmaY=max(1.0, roi.shape[0] / 18.0),
    )

    tamper_mask = np.power(tamper_mask * face_mask, 0.6)
    if float(tamper_mask.max()) <= 0.02:
        return False

    colored_heatmap = np.zeros((*resized_heatmap.shape, 3), dtype=np.uint8)
    colored_heatmap[..., 2] = 255
    colored_heatmap[..., 1] = np.uint8(np.clip((1.0 - resized_heatmap) * 255, 0, 255))
    alpha = np.clip(tamper_mask * 0.85, 0.0, 0.85)[..., None]
    blended_roi = roi.astype(np.float32) * (1.0 - alpha) + colored_heatmap.astype(np.float32) * alpha
    frame[top:bottom, left:right] = np.clip(blended_roi, 0, 255).astype(np.uint8)
    return True

def get_heatmap_for_frame(frame_index, sampled_frame_indices, heatmaps):
    if not heatmaps:
        return None

    usable_count = min(len(sampled_frame_indices), len(heatmaps))
    if usable_count <= 0:
        return heatmaps[0]

    indices = sampled_frame_indices[:usable_count]
    insert_position = int(np.searchsorted(indices, frame_index))
    if insert_position <= 0:
        selected_index = 0
    elif insert_position >= usable_count:
        selected_index = usable_count - 1
    else:
        previous_index = insert_position - 1
        next_index = insert_position
        if abs(indices[next_index] - frame_index) < abs(indices[previous_index] - frame_index):
            selected_index = next_index
        else:
            selected_index = previous_index
    return heatmaps[selected_index]

def get_nearest_sample_with_face(samples, frame_index):
    candidates = [
        sample
        for sample in samples
        if sample.face_location is not None
        and sample.frame_bgr is not None
        and sample.frame_bgr.size > 0
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda sample: abs(int(sample.frame_index) - int(frame_index)))

def scale_sample_face_to_preview(sample, preview_frame):
    if sample is None or sample.face_location is None or preview_frame is None:
        return None
    source_frame = ensure_bgr_frame(sample.frame_bgr)
    if source_frame is None:
        return None
    scale_x = preview_frame.shape[1] / float(max(1, source_frame.shape[1]))
    scale_y = preview_frame.shape[0] / float(max(1, source_frame.shape[0]))
    return scale_face_location(sample.face_location, scale_x, scale_y)

def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"

def sample_video_preview_frames(video_path, count):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total_frames > 0:
        frame_indices = np.linspace(0, max(0, total_frames - 1), num=count, dtype=int)
        for frame_index in sorted(set(int(index) for index in frame_indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                frames.append(frame)
            if len(frames) >= count:
                break
    else:
        while cap.isOpened() and len(frames) < count:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is not None and frame.size > 0:
                frames.append(frame)

    cap.release()
    return frames

def sample_video_preview_frames_with_indices(video_path, count):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total_frames > 0:
        frame_indices = np.linspace(0, max(0, total_frames - 1), num=count, dtype=int)
        for frame_index in sorted(set(int(index) for index in frame_indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                frames.append((frame_index, frame))
            if len(frames) >= count:
                break
    else:
        frame_index = 0
        while cap.isOpened() and len(frames) < count:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is not None and frame.size > 0:
                frames.append((frame_index, frame))
            frame_index += 1

    cap.release()
    return frames

class Model(nn.Module):

    def __init__(self, num_classes,latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        try:
            model = models.resnext50_32x4d(weights=None)
        except TypeError:
            model = models.resnext50_32x4d(pretrained=False)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048,num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        if seq_length > 1 and torch.equal(x, x[:, :1].expand_as(x)):
            x_first = x[:, 0, :, :, :]
            fmap_first = self.model(x_first)
            x_avg = self.avgpool(fmap_first)
            x_seq = x_avg.view(batch_size, 1, 2048).expand(batch_size, seq_length, 2048).contiguous()
            x_lstm,_ = self.lstm(x_seq,None)
            fmap = fmap_first.repeat_interleave(seq_length, dim=0)
            return fmap,self.dp(self.linear1(x_lstm[:,-1,:]))

        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,2048)
        x_lstm,_ = self.lstm(x,None)
        return fmap,self.dp(self.linear1(x_lstm[:,-1,:]))


ALLOWED_VIDEO_EXTENSIONS = set(['mp4','gif','webm','avi','3gp','wmv','flv','mkv'])
ALLOWED_IMAGE_EXTENSIONS = set(['jpg', 'jpeg', 'png', 'bmp', 'tiff'])
DEFAULT_VIDEO_SEQUENCE_LENGTH = 20
DEFAULT_IMAGE_SEQUENCE_LENGTH = 60

def get_file_extension(filename):
    if not filename or "." not in filename:
        return ""
    return filename.rsplit('.', 1)[1].lower()

def is_image_file(filename):
    return get_file_extension(filename) in ALLOWED_IMAGE_EXTENSIONS

def is_video_file(filename):
    return get_file_extension(filename) in ALLOWED_VIDEO_EXTENSIONS

def allowed_file(filename):
    ext = get_file_extension(filename)
    return ext in ALLOWED_VIDEO_EXTENSIONS or ext in ALLOWED_IMAGE_EXTENSIONS

def index(request):
    if request.method == 'GET':
        video_upload_form = VideoUploadForm()
        if 'file_name' in request.session:
            del request.session['file_name']
        if 'preprocessed_images' in request.session:
            del request.session['preprocessed_images']
        if 'faces_cropped_images' in request.session:
            del request.session['faces_cropped_images']
        if 'use_gemini_review' in request.session:
            del request.session['use_gemini_review']
        return render(request, index_template_name, {"form": video_upload_form})
    else:
        video_upload_form = VideoUploadForm(request.POST, request.FILES)
        if video_upload_form.is_valid():
            video_file = video_upload_form.cleaned_data['upload_video_file']
            video_file_ext = get_file_extension(video_file.name)
            sequence_length = video_upload_form.cleaned_data['sequence_length']
            use_gemini_review = bool(video_upload_form.cleaned_data.get('use_gemini_review'))

            if sequence_length <= 0:
                video_upload_form.add_error("sequence_length", "Sequence Length must be greater than 0")
                return render(request, index_template_name, {"form": video_upload_form})
            
            if allowed_file(video_file.name) == False:
                video_upload_form.add_error("upload_video_file","Only video and image files are allowed ")
                return render(request, index_template_name, {"form": video_upload_form})

            if is_image_file(video_file.name):
                sequence_length = DEFAULT_IMAGE_SEQUENCE_LENGTH
            elif is_video_file(video_file.name):
                sequence_length = DEFAULT_VIDEO_SEQUENCE_LENGTH
            
            saved_video_file = 'uploaded_file_'+str(int(time.time()))+"."+video_file_ext
            if settings.DEBUG:
                with open(os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file), 'wb') as vFile:
                    shutil.copyfileobj(video_file, vFile)
                request.session['file_name'] = os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file)
            else:
                with open(os.path.join(settings.PROJECT_DIR, 'uploaded_videos','app','uploaded_videos', saved_video_file), 'wb') as vFile:
                    shutil.copyfileobj(video_file, vFile)
                request.session['file_name'] = os.path.join(settings.PROJECT_DIR, 'uploaded_videos','app','uploaded_videos', saved_video_file)
            request.session['sequence_length'] = sequence_length
            request.session['use_gemini_review'] = use_gemini_review
            return redirect('ml_app:predict')
        else:
            return render(request, index_template_name, {"form": video_upload_form})

def predict_page(request):
    if request.method == "GET":
        # Redirect to 'home' if 'file_name' is not in session
        if 'file_name' not in request.session:
            return redirect("ml_app:home")
        if 'file_name' in request.session:
            video_file = request.session['file_name']
        if 'sequence_length' in request.session:
            sequence_length = request.session['sequence_length']
        use_gemini_review = bool(request.session.get('use_gemini_review', False))
        path_to_videos = [video_file]
        video_file_name = os.path.basename(video_file)
        video_file_name_only = os.path.splitext(video_file_name)[0]
        # Production environment adjustments
        if not settings.DEBUG:
            production_video_name = os.path.join('/home/app/staticfiles/', video_file_name.split('/')[3])
            print("Production file name", production_video_name)
        else:
            production_video_name = video_file_name
        result_video_url = None

        start_time = time.time()
        is_static_image = video_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
        media_lane = "image" if is_static_image else "video"
        if not is_static_image:
            result_video_url = f"{settings.MEDIA_URL}{video_file_name}"

        try:
            from .hybrid_inference import analyze_media, select_tracked_face
            analysis = analyze_media(video_file, media_lane, review_required_override=use_gemini_review)
        except Exception as e:
            print(f"Exception occurred during prediction: {e}")
            return render(
                request,
                index_template_name,
                {
                    "form": VideoUploadForm(),
                    "prediction_error": f"Prediction failed: {e}",
                },
            )

        sequence_length = analysis.local_result.sequence_length
        model_name = analysis.checkpoint_name

        print("<=== | Started Media Preview Export | ===>")
        preprocessed_images = []
        faces_cropped_images = []
        result_preview_image = None
        result_preview_frames = []
        result_media_duration = "0:00"
        result_media_duration_seconds = 0
        faces_found = 0
        samples = analysis.local_result.samples[:sequence_length]
        if is_static_image and len(samples) == 1:
            samples = samples * sequence_length

        if not is_static_image:
            duration_cap = cv2.VideoCapture(video_file)
            if duration_cap.isOpened():
                fps = duration_cap.get(cv2.CAP_PROP_FPS)
                frame_count = duration_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if fps and fps > 0 and frame_count and frame_count > 0:
                    result_media_duration_seconds = frame_count / fps
                    result_media_duration = format_duration(result_media_duration_seconds)
            duration_cap.release()

            preview_fps = get_detection_float_setting("WEB_RESULT_PREVIEW_FPS", 10.0)
            preview_max_frames = get_detection_int_setting("WEB_RESULT_PREVIEW_MAX_FRAMES", 140)
            preview_frame_count = max(
                20,
                min(preview_max_frames, int(max(1.0, result_media_duration_seconds) * preview_fps)),
            )
            preview_source_frames = sample_video_preview_frames_with_indices(video_file, preview_frame_count)
            previous_preview_face = None
            face_detect_interval = max(1, get_detection_int_setting("WEB_RESULT_FACE_DETECT_INTERVAL", 3))
            sample_face_max_distance = max(1, get_detection_int_setting("WEB_RESULT_SAMPLE_FACE_MAX_DISTANCE", 12))

            for preview_index, (frame_index, source_preview) in enumerate(preview_source_frames):
                preview_frame = resize_frame_for_preview(source_preview, max_dimension=960)
                if preview_frame is None:
                    continue

                selected_face = None
                nearest_sample = get_nearest_sample_with_face(
                    analysis.local_result.samples,
                    frame_index,
                )
                if (
                    nearest_sample is not None
                    and abs(int(nearest_sample.frame_index) - int(frame_index)) <= sample_face_max_distance
                ):
                    selected_face = scale_sample_face_to_preview(nearest_sample, preview_frame)
                    if selected_face is not None:
                        previous_preview_face = selected_face

                if selected_face is None and (preview_index % face_detect_interval == 0 or previous_preview_face is None):
                    detected_faces = detect_face_locations(preview_frame)
                    selected_face = select_tracked_face(detected_faces, previous_preview_face)
                    if selected_face is not None:
                        previous_preview_face = selected_face
                elif selected_face is None:
                    selected_face = previous_preview_face

                if selected_face is None:
                    continue

                heatmap = get_heatmap_for_frame(
                    frame_index,
                    analysis.local_result.sampled_frame_indices,
                    analysis.local_result.heatmaps,
                )
                if analysis.output_should_overlay:
                    apply_tamper_overlay(preview_frame, selected_face, heatmap, analysis.label)
                draw_face_read_box(preview_frame, selected_face, analysis.label, analysis.confidence)

                preview_rgb = ensure_rgb_frame(preview_frame)
                if preview_rgb is None:
                    continue

                preview_name = f"{video_file_name_only}_result_preview_{len(result_preview_frames)+1}.png"
                preview_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', preview_name)
                pImage.fromarray(preview_rgb).save(preview_path)
                result_preview_frames.append(preview_name)
                if result_preview_image is None:
                    result_preview_image = preview_name

        print(f"Number of sampled frames: {len(samples)}")
        for i, sample in enumerate(samples):
            source_frame = ensure_bgr_frame(sample.frame_bgr)
            frame = resize_frame_for_preview(source_frame)
            if frame is None:
                continue

            rgb_frame = ensure_rgb_frame(frame)
            if rgb_frame is None:
                continue

            image_name = f"{video_file_name_only}_preprocessed_{i+1}.png"
            image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
            img_rgb = pImage.fromarray(rgb_frame)
            img_rgb.save(image_path)
            preprocessed_images.append(image_name)

            if is_static_image and sample.face_location is not None and source_frame is not None:
                annotated_frame = frame.copy()
                scale_x = annotated_frame.shape[1] / float(max(1, source_frame.shape[1]))
                scale_y = annotated_frame.shape[0] / float(max(1, source_frame.shape[0]))
                preview_face = scale_face_location(sample.face_location, scale_x, scale_y)
                if preview_face is not None:
                    heatmap = get_heatmap_for_frame(
                        sample.frame_index,
                        analysis.local_result.sampled_frame_indices,
                        analysis.local_result.heatmaps,
                    )
                    if analysis.output_should_overlay:
                        apply_tamper_overlay(annotated_frame, preview_face, heatmap, analysis.label)
                    draw_face_read_box(annotated_frame, preview_face, analysis.label, analysis.confidence)
                    annotated_rgb = ensure_rgb_frame(annotated_frame)
                    if annotated_rgb is not None:
                        preview_name = f"{video_file_name_only}_result_preview_{len(result_preview_frames)+1}.png"
                        preview_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', preview_name)
                        pImage.fromarray(annotated_rgb).save(preview_path)
                        result_preview_frames.append(preview_name)
                        if result_preview_image is None:
                            result_preview_image = preview_name

            if (
                not sample.used_face
                or sample.face_location is None
                or sample.face_crop_bgr is None
                or sample.face_crop_bgr.size == 0
                or analysis.local_result.valid_face_crops == 0
            ):
                continue

            face_preview = resize_frame_for_preview(sample.face_crop_bgr, max_dimension=420)
            rgb_face = ensure_rgb_frame(face_preview)
            if rgb_face is None:
                continue

            img_face_rgb = pImage.fromarray(rgb_face)
            image_name = f"{video_file_name_only}_cropped_faces_{i+1}.png"
            image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
            img_face_rgb.save(image_path)
            faces_found += 1
            faces_cropped_images.append(image_name)

        print("<=== | Media Preview Export Done | ===>")
        print("--- %s seconds ---" % (time.time() - start_time))

        # No face detected
        if faces_found == 0:
            return render(request, predict_template_name, {"no_faces": True})

        print("<=== | Prediction Done | ===>")
        print(
            "Prediction:",
            analysis.label,
            "Confidence:",
            analysis.confidence,
            "Model:",
            model_name,
        )
        print("--- %s seconds ---" % (time.time() - start_time))

        context = {
            'preprocessed_images': preprocessed_images,
            'faces_cropped_images': faces_cropped_images,
            'heatmap_images': [],
            'original_video': production_video_name,
            'models_location': os.path.join(settings.PROJECT_DIR, 'models'),
            'model_used': model_name,
            'output': analysis.label,
            'confidence': analysis.confidence,
            'decision_source': analysis.decision_source,
            'gemini_review_enabled': use_gemini_review,
            'result_preview_image': result_preview_image,
            'result_preview_frames': result_preview_frames,
            'result_preview_interval_ms': max(
                60,
                int((result_media_duration_seconds / max(1, len(result_preview_frames))) * 1000)
                if result_media_duration_seconds and result_preview_frames
                else 120,
            ),
            'show_result_player': not is_static_image and len(result_preview_frames) > 1,
            'result_video_url': result_video_url,
            'result_media_duration': result_media_duration,
            'result_media_duration_seconds': result_media_duration_seconds,
            'gemini_status': analysis.cloud_result.status,
            'gemini_label': analysis.cloud_result.label,
            'gemini_confidence': analysis.cloud_result.confidence,
            'gemini_provider': analysis.cloud_result.provider,
        }

        if settings.DEBUG:
            return render(request, predict_template_name, context)
        else:
            return render(request, predict_template_name, context)
def about(request):
    return render(request, about_template_name)

def handler404(request,exception):
    return render(request, '404.html', status=404)
def cuda_full(request):
    return render(request, 'cuda_full.html')
