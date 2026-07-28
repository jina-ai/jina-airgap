import base64
import io
from typing import Union

from errors import BadRequest, PayloadTooLarge

MAX_MEDIA_BYTES = 10 * 1024 * 1024  # 10 MB per input

IMAGE_MIMES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/gif",
    "image/bmp",
    "image/tiff",
    "image/avif",
    "image/heic",
    "image/svg+xml",
}
AUDIO_MIMES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mp3",
    "audio/mpeg",
    "audio/flac",
    "audio/ogg",
    "audio/m4a",
    "audio/x-m4a",
    "audio/opus",
}
VIDEO_MIMES = {
    "video/mp4",
    "video/avi",
    "video/x-msvideo",
    "video/quicktime",
    "video/x-matroska",
    "video/webm",
    "video/x-flv",
    "video/x-ms-wmv",
}


def _decode_b64(b64_str: str) -> tuple:
    """
    Decode a base64 string in raw or data-URL format.
    Returns (raw_bytes, mime_type).

    Accepts:
    - Raw base64: "iVBORw0KGgo..."
    - Data URL:   "data:image/png;base64,iVBORw0KGgo..."
    """
    mime_type = ""
    data = b64_str.strip()
    if data.startswith("data:"):
        try:
            header, data = data.split(",", 1)
            mime_type = header.split(";")[0][5:]  # strip "data:"
        except ValueError:
            raise BadRequest("Malformed data URL")
    try:
        raw = base64.b64decode(data)
    except Exception as e:
        raise BadRequest(f"Invalid base64 encoding: {e}")
    if len(raw) > MAX_MEDIA_BYTES:
        raise PayloadTooLarge(
            f"Media too large: {len(raw):,} bytes exceeds {MAX_MEDIA_BYTES:,} byte limit"
        )
    return raw, mime_type


def _detect_mime(raw: bytes, hint: str = "") -> str:
    """Detect MIME type from magic bytes; fall back to hint."""
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:2] == b"\xff\xd8":
        return "image/jpeg"
    if len(raw) >= 6 and raw[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return "audio/wav"
    if len(raw) >= 3 and raw[:3] == b"ID3":
        return "audio/mp3"
    if len(raw) >= 4 and raw[:4] == b"fLaC":
        return "audio/flac"
    if len(raw) >= 12 and b"ftyp" in raw[4:12]:
        return "video/mp4"
    if len(raw) >= 4 and raw[:4] == b"\x1a\x45\xdf\xa3":
        return "video/webm"
    return hint


def _bytes_to_st_input(raw: bytes, mime_hint: str = ""):
    """
    Convert raw bytes to the input type expected by sentence-transformers encode().
    - image/* -> PIL.Image.Image
    - audio/* or video/* -> io.BytesIO (sentence-transformers accepts bytes/BytesIO)
    """
    mime = (_detect_mime(raw, mime_hint) or mime_hint).lower()

    if mime in IMAGE_MIMES:
        from PIL import Image

        try:
            img = Image.open(io.BytesIO(raw))
            img.load()
            return img
        except Exception as e:
            raise BadRequest(f"Cannot decode image: {e}")

    if mime in AUDIO_MIMES or mime in VIDEO_MIMES:
        return io.BytesIO(raw)

    # Unknown MIME: attempt image decode, fall back to BytesIO
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(raw))
        img.load()
        return img
    except Exception:
        return io.BytesIO(raw)


def _parse_typed_base64(item: dict, key: str, default_mime: str) -> list:
    """Parse image_base64 / audio_base64 / video_base64 typed fields."""
    inner = item.get(key, "")
    if isinstance(inner, str):
        raw, mime = _decode_b64(inner)
        mime = mime or default_mime
    elif isinstance(inner, dict):
        raw, mime = _decode_b64(inner.get("base64", inner.get("data", "")))
        mime = mime or inner.get("mime_type", inner.get("mimeType", default_mime))
    else:
        raise BadRequest(f"Invalid value for '{key}'")
    return [_bytes_to_st_input(raw, mime)]


def _parse_content_part(part: dict) -> list:
    """
    Parse a single content-array part into a list of ST-compatible inputs.

    Handles:
    - {"type": "text", "text": "..."}
    - {"type": "image", "format": "base64", "value": "..."}         (Elastic format)
    - {"type": "image_url", "image_url": {"url": "data:..."}}       (Cohere/Voyage)
    - {"type": "image_base64", "image_base64": "data:..." | {...}}
    - {"type": "audio_base64", "audio_base64": "data:..." | {...}}
    - {"type": "video_base64", "video_base64": "data:..." | {...}}
    - {"inlineData": {"mimeType": "...", "data": "..."}}             (Gemini format)
    """
    if not isinstance(part, dict):
        raise BadRequest(f"Content part must be a dict, got {type(part).__name__}")

    t = part.get("type", "")

    # `{"type": "text", ...}` is OpenAI's spelling; a bare `{"text": ...}` with
    # no discriminator is Jina's own TextDoc, which the public API accepts in
    # `input` and which `documents` already accepts on the rerank side.
    if t == "text" or (not t and "text" in part):
        return [part.get("text", "")]

    # Elastic Inference Service: {"type": "image", "format": "base64", "value": "..."}
    if t == "image" and part.get("format") == "base64":
        raw, mime = _decode_b64(part["value"])
        return [_bytes_to_st_input(raw, mime or "image/jpeg")]

    # Cohere/Voyage: {"type": "image_url", "image_url": {"url": "data:..."}}
    if t == "image_url":
        url_val = part.get("image_url", {})
        url = url_val.get("url", url_val) if isinstance(url_val, dict) else str(url_val)
        if not url.startswith("data:"):
            raise BadRequest(
                "image_url: only data: URLs (base64) are supported in air-gapped mode"
            )
        raw, mime = _decode_b64(url)
        return [_bytes_to_st_input(raw, mime)]

    # Voyage documents a video_url part. Fetching it needs egress, so name the
    # reason rather than let it fall through to "unknown part type".
    if t == "video_url":
        raise BadRequest(
            "video_url: only data: URLs (base64) are supported in air-gapped mode"
        )

    # Gemini: {"inlineData": {"mimeType": "...", "data": "..."}}
    if "inlineData" in part:
        inline = part["inlineData"]
        raw, _ = _decode_b64(inline.get("data", ""))
        return [_bytes_to_st_input(raw, inline.get("mimeType", ""))]

    # Typed base64 formats
    if t == "image_base64" or "image_base64" in part:
        return _parse_typed_base64(part, "image_base64", "image/jpeg")
    if t == "audio_base64" or "audio_base64" in part:
        return _parse_typed_base64(part, "audio_base64", "audio/wav")
    if t == "video_base64" or "video_base64" in part:
        return _parse_typed_base64(part, "video_base64", "video/mp4")

    raise BadRequest(f"Unknown content part type: '{t}'")


def _parse_openai_item(item) -> list:
    """
    Parse one element of the OpenAI `input` array into a list of ST-compatible inputs.

    Returns a list:
    - [str]          -> plain text (len 1)
    - [PIL.Image]    -> single image (len 1)
    - [BytesIO]      -> single audio/video (len 1)
    - [x, y, ...]    -> fused multimodal parts (len > 1, caller wraps in tuple)

    Supported formats:
    1. "plain text"
    2. {"type": "text", "text": "..."}
    3. {"type": "image", "format": "base64", "value": "..."}                    (Elastic)
    4. {"type": "image_base64", "image_base64": {"base64": "...", "mime_type": "..."}}
    5. {"type": "audio_base64", "audio_base64": {"base64": "...", "mime_type": "..."}}
    6. {"type": "video_base64", "video_base64": {"base64": "...", "mime_type": "..."}}
    7. {"content": [...]}  fused multimodal block -> parts merged into ONE embedding
    """
    if isinstance(item, str):
        return [item]

    if not isinstance(item, dict):
        raise BadRequest(f"Input item must be str or dict, got {type(item).__name__}")

    # Fused content block: {"content": [...]}
    if "content" in item and isinstance(item["content"], list):
        parts = []
        for p in item["content"]:
            parts.extend(_parse_content_part(p))
        return parts

    # Single-part item: delegate to content-part parser
    return _parse_content_part(item)


def fuse_content(content: list) -> Union[str, list]:
    """Flatten a content-part array into one fused embedding input.

    A list of parts means one embedding over all of them, not one embedding
    each -- Cohere's `inputs[].content`, Voyage's `inputs[].content` and
    Gemini's `content.parts` all mean the same thing.
    """
    parts = []
    for part in content:
        parts.extend(_parse_content_part(part))
    if len(parts) > 1:
        return parts
    return parts[0] if parts else ""
