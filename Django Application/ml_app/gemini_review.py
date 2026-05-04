import base64
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
from django.conf import settings
from urllib import error as urllib_error
from urllib import request as urllib_request


DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"
DEFAULT_GEMINI_FALLBACK_MODELS = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]
DEFAULT_GEMINI_TIMEOUT_SECONDS = 20


@dataclass
class GeminiReviewImage:
    label: str
    image_bgr: object


@dataclass
class GeminiReviewResult:
    label: str = "REAL"
    confidence: float = 0.0
    reason: str = ""
    suspicious_frames: List[str] = field(default_factory=list)
    provider: str = "Gemini"
    status: str = "skipped"
    raw_response_text: str = ""


def _get_bool_setting(name: str, default: bool) -> bool:
    value = getattr(settings, name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _get_float_setting(name: str, default: float) -> float:
    value = getattr(settings, name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_list_setting(name: str, default: List[str]) -> List[str]:
    value = getattr(settings, name, None) or os.getenv(name, "")
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",")]
    else:
        values = [str(item).strip() for item in value]
    values = [item for item in values if item]
    return values or list(default)


def _compact_http_error(detail: str) -> str:
    try:
        payload = json.loads(detail)
        message = payload.get("error", {}).get("message", "")
        if message:
            return " ".join(message.split())[:220]
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    return " ".join((detail or "").split())[:220]


def _encode_image_to_base64(image_bgr) -> Optional[str]:
    success, buffer = cv2.imencode(".jpg", image_bgr)
    if not success:
        return None
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _extract_json_payload(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1:
        cleaned = cleaned[first_brace:last_brace + 1]
    return json.loads(cleaned)


def _parse_confidence_value(raw_confidence) -> float:
    if isinstance(raw_confidence, str):
        raw_confidence = raw_confidence.strip().rstrip("%")

    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        return 0.0

    if 0.0 <= confidence <= 1.0:
        confidence *= 100.0

    return max(0.0, min(100.0, confidence))


def review_face_crops(
    media_lane: str,
    review_images: List[GeminiReviewImage],
    fake_probability: float,
) -> GeminiReviewResult:
    enabled = _get_bool_setting("ENABLE_GEMINI_REVIEW", True)
    if not enabled:
        return GeminiReviewResult(status="skipped", reason="Gemini review disabled in settings")

    api_key = getattr(settings, "GEMINI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return GeminiReviewResult(status="skipped", reason="Missing GEMINI_API_KEY")

    if not review_images:
        return GeminiReviewResult(status="skipped", reason="No review crops available")

    model_name = getattr(settings, "GEMINI_MODEL", DEFAULT_GEMINI_MODEL) or DEFAULT_GEMINI_MODEL
    fallback_models = _get_list_setting("GEMINI_FALLBACK_MODELS", DEFAULT_GEMINI_FALLBACK_MODELS)
    model_names = []
    for candidate_model in [model_name, *fallback_models]:
        if candidate_model and candidate_model not in model_names:
            model_names.append(candidate_model)

    timeout_seconds = _get_float_setting("GEMINI_TIMEOUT_SECONDS", DEFAULT_GEMINI_TIMEOUT_SECONDS)

    prompt_lines = [
        "You are reviewing media from a local deepfake detector.",
        "For this app, FAKE means AI-generated/synthetic media, deepfake media, face manipulation, or generated portrait content.",
        "REAL means a genuine camera capture with no visible AI generation or manipulation evidence.",
        f"The local detector ran in the '{media_lane}' lane and estimated fake probability {fake_probability:.3f}.",
        f"You are receiving {len(review_images)} labeled review images from the media.",
        "Review every provided image before deciding.",
        "Some images are full-frame context or corner crops so you can inspect watermarks, object/text errors, hands, lighting, and generator branding.",
        "Look for visible AI watermarks, labels, logos, badges, or signatures such as Gemini sparkle/star marks, Sora, Veo, Runway, Pika, Kling, SynthID, or similar generator marks.",
        "A clearly visible generator watermark, AI badge, or generator logo is strong evidence for FAKE.",
        "Visible synthetic-image clues such as malformed text, impossible object geometry, unnatural hands, overly generated skin/hair, or inconsistent lighting are evidence for FAKE.",
        "Do not mark a normal camera photo as FAKE only because clothing/background text is blurred, compressed, partially hidden, or hard to read.",
        "Camera timestamps, low resolution, compression noise, pose blur, and difficult crop angles are normal REAL-photo artifacts unless there are multiple strong AI signs.",
        "If a provided crop does not clearly contain a face, rely more on the full-frame context and avoid using that crop alone as FAKE evidence.",
        "Do not invent or assume a watermark when none is visible.",
        "Decide whether the media shows signs of AI generation, face tampering, deepfake artifacts, or generator watermarking.",
        "Return JSON only with this exact shape:",
        '{"label":"REAL or FAKE","confidence":95,"reason":"short reason","suspicious_frames":["Crop 1"]}',
        "Confidence must be a number from 0 to 100, not a 0 to 1 fraction.",
        "Use REAL only when the full context and face crop show no visible evidence of AI generation or manipulation.",
    ]

    parts = [{"text": "\n".join(prompt_lines)}]
    for review_image in review_images:
        encoded_image = _encode_image_to_base64(review_image.image_bgr)
        if not encoded_image:
            continue
        parts.append({"text": review_image.label})
        parts.append(
            {
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": encoded_image,
                }
            }
        )

    if len(parts) == 1:
        return GeminiReviewResult(status="skipped", reason="Could not encode review crops")

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }

    request_body = json.dumps(payload).encode("utf-8")
    errors = []

    for candidate_model in model_names:
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{candidate_model}:generateContent?key={api_key}"
        )
        request = urllib_request.Request(
            endpoint,
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            candidates = response_data.get("candidates") or []
            if not candidates:
                errors.append(f"{candidate_model}: no candidates")
                continue

            candidate_parts = (
                candidates[0].get("content", {}).get("parts", [])
                if isinstance(candidates[0], dict)
                else []
            )
            response_text = ""
            for part in candidate_parts:
                if isinstance(part, dict) and part.get("text"):
                    response_text += part["text"]
            if not response_text.strip():
                errors.append(f"{candidate_model}: empty text")
                continue

            parsed = _extract_json_payload(response_text)
            label = str(parsed.get("label", "REAL")).strip().upper()
            if label not in {"REAL", "FAKE"}:
                label = "REAL"

            confidence = _parse_confidence_value(parsed.get("confidence", 0.0))

            reason = str(parsed.get("reason", "")).strip()
            suspicious_frames = parsed.get("suspicious_frames", [])
            if not isinstance(suspicious_frames, list):
                suspicious_frames = []
            suspicious_frames = [str(item).strip() for item in suspicious_frames if str(item).strip()]

            return GeminiReviewResult(
                label=label,
                confidence=confidence,
                reason=reason,
                suspicious_frames=suspicious_frames,
                provider=f"Gemini ({candidate_model})",
                status="reviewed",
                raw_response_text=response_text,
            )
        except TimeoutError:
            errors.append(f"{candidate_model}: timed out")
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            errors.append(f"{candidate_model}: HTTP {exc.code} {_compact_http_error(detail)}")
        except urllib_error.URLError as exc:
            errors.append(f"{candidate_model}: request failed {exc}")
            break
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{candidate_model}: response parse failed {exc}")

    return GeminiReviewResult(
        status="error",
        reason="Gemini fallback failed: " + "; ".join(errors),
    )
