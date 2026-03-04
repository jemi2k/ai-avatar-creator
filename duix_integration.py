"""
DuiX Avatar Video Synthesis integration for IndexTTS WebUI.

Docker bind mount:
  HOST:       D:\\duix_avatar_data\\face2face
  CONTAINER:  /code/data

The DuiX gen-video API returns result paths like:
  "/762e9890-...-r.mp4"  (leading slash + bare filename)

The Electron app resolves via: path.join(assetPath.model, result)
  assetPath.model = "D:\\duix_avatar_data\\face2face\\temp"
  → "D:\\duix_avatar_data\\face2face\\temp\\762e9890-...-r.mp4"

Features:
  - Audio extraction from uploaded video (via ffmpeg/ffprobe)
  - Copying files into the DuiX shared volume
  - Submitting video synthesis jobs to the DuiX face2face API
  - Polling for job completion
  - Resolving container-relative result path back to host path
  - Avatar library (save/load/delete video+audio pairs)
  - AI prompt generation via DeepSeek API

Prerequisites:
  - ffmpeg + ffprobe on PATH
  - pip install requests
  - DuiX gen-video Docker container running on port 8384
"""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
import uuid

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DUIX_BASE_URL   = os.environ.get("DUIX_API_URL", "http://127.0.0.1:8384/easy")
DUIX_SUBMIT_URL = f"{DUIX_BASE_URL}/submit"
DUIX_QUERY_URL  = f"{DUIX_BASE_URL}/query"

POLL_INTERVAL = 2.0
POLL_TIMEOUT  = 600

_is_win = platform.system() == "Windows"

DUIX_HOST_DATA_ROOT = os.environ.get(
    "DUIX_HOST_DATA_ROOT",
    os.path.join("D:\\", "duix_avatar_data", "face2face") if _is_win
    else os.path.join(os.path.expanduser("~"), "duix_avatar_data", "face2face"),
)
DUIX_HOST_TEMP = os.path.join(DUIX_HOST_DATA_ROOT, "temp")

DUIX_CONTAINER_DATA_ROOT = "/code/data"
DUIX_CONTAINER_TEMP      = f"{DUIX_CONTAINER_DATA_ROOT}/temp"


# ---------------------------------------------------------------------------
# Avatar library
# ---------------------------------------------------------------------------
AVATARS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "avatars")
AVATARS_JSON = os.path.join(AVATARS_DIR, "avatars.json")


def _ensure_avatars_dir():
    os.makedirs(AVATARS_DIR, exist_ok=True)


def _load_avatars() -> list[dict]:
    """Load the avatar list from disk."""
    _ensure_avatars_dir()
    if not os.path.isfile(AVATARS_JSON):
        return []
    try:
        with open(AVATARS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        return []


def _save_avatars(avatars: list[dict]):
    """Write the avatar list to disk."""
    _ensure_avatars_dir()
    with open(AVATARS_JSON, "w", encoding="utf-8") as f:
        json.dump(avatars, f, indent=2, ensure_ascii=False)


def _file_hash(path: str) -> str:
    """Quick MD5 of the first 64KB — enough to detect duplicates."""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            h.update(f.read(65536))
    except IOError:
        return ""
    return h.hexdigest()


def _copy_to_avatars(src_path: str, prefix: str) -> str:
    """Copy a file into the avatars/ folder. Returns the new path."""
    _ensure_avatars_dir()
    ext = os.path.splitext(src_path)[1]
    filename = f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:6]}{ext}"
    dst = os.path.join(AVATARS_DIR, filename)
    shutil.copy2(src_path, dst)
    return dst


def get_avatar_choices() -> list[str]:
    """Return list of avatar display names for a dropdown."""
    avatars = _load_avatars()
    return [a["name"] for a in avatars]


def load_avatar_by_name(name: str) -> dict | None:
    """Look up an avatar by display name. Returns dict or None."""
    avatars = _load_avatars()
    for a in avatars:
        if a["name"] == name:
            return a
    return None


def save_avatar(name: str, video_path: str, audio_path: str) -> dict:
    """
    Save a new avatar. Copies video+audio into avatars/ folder.
    Returns the saved avatar dict.
    Skips if an avatar with the same video+audio content already exists.
    """
    avatars = _load_avatars()

    # Check for duplicate by file content hash
    vid_hash = _file_hash(video_path)
    aud_hash = _file_hash(audio_path)

    for a in avatars:
        existing_vid_hash = _file_hash(a.get("video_path", ""))
        existing_aud_hash = _file_hash(a.get("audio_path", ""))
        if existing_vid_hash == vid_hash and existing_aud_hash == aud_hash:
            print(f"[Avatar] Duplicate found: '{a['name']}' — skipping save")
            return a

    # Copy files to persistent storage
    saved_video = _copy_to_avatars(video_path, "video")
    saved_audio = _copy_to_avatars(audio_path, "audio")

    avatar = {
        "id": uuid.uuid4().hex[:8],
        "name": name,
        "video_path": saved_video,
        "audio_path": saved_audio,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    avatars.append(avatar)
    _save_avatars(avatars)
    print(f"[Avatar] Saved: '{name}' (video={saved_video}, audio={saved_audio})")
    return avatar


def delete_avatar(name: str) -> bool:
    """Delete an avatar by name. Also removes its files."""
    avatars = _load_avatars()
    found = None
    for a in avatars:
        if a["name"] == name:
            found = a
            break
    if not found:
        return False

    # Remove files
    for key in ("video_path", "audio_path"):
        fp = found.get(key, "")
        if fp and os.path.isfile(fp):
            try:
                os.remove(fp)
                print(f"[Avatar] Deleted file: {fp}")
            except OSError as e:
                print(f"[Avatar] Could not delete {fp}: {e}")

    avatars = [a for a in avatars if a["name"] != name]
    _save_avatars(avatars)
    print(f"[Avatar] Deleted avatar: '{name}'")
    return True


# ---------------------------------------------------------------------------
# Shared volume helpers
# ---------------------------------------------------------------------------

def _ensure_shared_dir():
    os.makedirs(DUIX_HOST_TEMP, exist_ok=True)


def _copy_to_shared_volume(local_path: str, prefix: str = "") -> tuple[str, str]:
    _ensure_shared_dir()
    ext = os.path.splitext(local_path)[1]
    filename = f"{prefix}{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"
    host_dst = os.path.join(DUIX_HOST_TEMP, filename)
    shutil.copy2(local_path, host_dst)
    container_path = f"{DUIX_CONTAINER_TEMP}/{filename}"
    print(f"[DuiX] Copied {local_path}  →  {host_dst}  (container: {container_path})")
    return host_dst, container_path


def _container_result_to_host(result_value: str) -> str:
    """
    Convert the DuiX API result path to a host-side file path.

    Real-world observed format from the API:
        "/762e9890-a101-43fa-bd05-a1f00a31a775-r.mp4"

    This is just a bare filename with a leading slash.
    The DuiX Electron app resolves it with:
        path.join("D:\\duix_avatar_data\\face2face\\temp", result)
    On Node.js, path.join strips the leading slash, so it becomes:
        "D:\\duix_avatar_data\\face2face\\temp\\762e9890-...-r.mp4"

    We replicate that same logic here.
    """
    if not result_value:
        return ""

    # Strip container-absolute prefix if present
    rel = result_value
    if rel.startswith(DUIX_CONTAINER_DATA_ROOT):
        rel = rel[len(DUIX_CONTAINER_DATA_ROOT):]

    # Strip any leading slashes
    rel = rel.lstrip("/").lstrip("\\")

    # If it starts with "temp/" strip that since DUIX_HOST_TEMP already includes /temp
    if rel.startswith("temp/") or rel.startswith("temp\\"):
        rel = rel.split("/", 1)[-1] if "/" in rel else rel.split("\\", 1)[-1]

    # Everything ends up in the temp directory
    host_path = os.path.join(DUIX_HOST_TEMP, rel)

    print(f"[DuiX] Result path mapping: '{result_value}' → '{host_path}'")
    return host_path


# ---------------------------------------------------------------------------
# 1. Audio extraction from video
# ---------------------------------------------------------------------------
def extract_audio_from_video(video_path: str) -> str | None:
    if not video_path or not os.path.isfile(video_path):
        return None

    output_dir = os.path.join("outputs", "extracted_audio")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"extracted_{int(time.time())}.wav")

    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=30,
        )
        if "audio" not in probe.stdout:
            print(f"[DuiX] No audio stream in {video_path}")
            return None

        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn",
             "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path],
            capture_output=True, text=True, timeout=120, check=True,
        )
        if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
            print(f"[DuiX] Extracted audio → {output_path}")
            return output_path
        return None
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[DuiX] Audio extraction failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# 2. Submit synthesis job
# ---------------------------------------------------------------------------
def submit_video_synthesis(audio_path: str, video_path: str) -> dict:
    task_code = str(uuid.uuid4())
    try:
        _, container_audio = _copy_to_shared_volume(audio_path, prefix="audio_")
        _, container_video = _copy_to_shared_volume(video_path, prefix="video_")
    except Exception as exc:
        return {"success": False, "task_code": task_code,
                "message": f"Failed to copy files to shared volume: {exc}"}

    payload = {
        "audio_url": container_audio,
        "video_url": container_video,
        "code": task_code,
        "chaofen": 0,
        "watermark_switch": 0,
        "pn": 1,
    }
    print(f"[DuiX] Submitting: {payload}")
    try:
        resp = requests.post(DUIX_SUBMIT_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") == 10000:
            return {"success": True, "task_code": task_code,
                    "message": "Job submitted successfully"}
        return {"success": False, "task_code": task_code,
                "message": f"DuiX API error: {data.get('msg', 'Unknown')}"}
    except requests.RequestException as exc:
        return {"success": False, "task_code": task_code,
                "message": f"Cannot reach DuiX API at {DUIX_SUBMIT_URL}: {exc}"}


# ---------------------------------------------------------------------------
# 3. Poll for result
# ---------------------------------------------------------------------------
def poll_video_result(task_code: str, progress_callback=None) -> dict:
    t0 = time.time()
    while True:
        if time.time() - t0 > POLL_TIMEOUT:
            return {"success": False, "video_path": None,
                    "message": f"Timed out after {POLL_TIMEOUT}s"}
        try:
            resp = requests.get(DUIX_QUERY_URL, params={"code": task_code}, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            print(f"[DuiX] Poll response: {data}")

            if data.get("code") in (9999, 10002, 10003):
                return {"success": False, "video_path": None,
                        "message": f"DuiX error: {data.get('msg', 'Unknown')}"}
            if data.get("code") == 10000:
                inner = data.get("data", {})
                status = inner.get("status")
                if status == 1:
                    pv = inner.get("progress", 0)
                    msg = inner.get("msg", "Processing…")
                    if progress_callback:
                        try:
                            progress_callback(
                                pv / 100.0 if isinstance(pv, (int, float)) else 0.5, msg)
                        except Exception:
                            pass
                elif status == 2:
                    raw_result = inner.get("result", "")
                    host_result = _container_result_to_host(raw_result)
                    print(f"[DuiX] Done! raw='{raw_result}' → host='{host_result}'")
                    print(f"[DuiX] File exists: {os.path.isfile(host_result)}")
                    return {"success": True, "video_path": host_result,
                            "message": "Synthesis complete"}
                elif status == 3:
                    return {"success": False, "video_path": None,
                            "message": f"Failed: {inner.get('msg', 'Unknown')}"}
        except requests.RequestException as exc:
            print(f"[DuiX] poll error (retrying): {exc}")
        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# 4. AI Prompt Generation (DeepSeek API)
# ---------------------------------------------------------------------------
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
DEEPSEEK_MODEL   = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

PROMPT_SYSTEM = """You are a script writer for talking-head avatar videos.
The user will give you a brief idea or topic. You must write a natural,
conversational script that sounds good when spoken aloud by a virtual avatar.

Rules:
- Write ONLY the script text, no stage directions, no quotes, no labels
- Keep it natural and conversational, as if speaking directly to the viewer
- Match the language of the user's input (Chinese input → Chinese output, English → English, etc.)
- Keep it concise: 1-3 paragraphs unless the user asks for more
- No emojis, no markdown formatting, just plain spoken text
"""


def generate_prompt_text(user_idea: str, api_key: str = "") -> dict:
    """
    Call DeepSeek API to generate a speaking script from a brief idea.

    Parameters
    ----------
    user_idea : str   The user's brief description / topic
    api_key   : str   Override API key (if empty, uses env var)

    Returns dict with  success, text, message
    """
    key = api_key or DEEPSEEK_API_KEY
    if not key:
        return {
            "success": False,
            "text": "",
            "message": "No DeepSeek API key. Set DEEPSEEK_API_KEY environment variable "
                       "or pass --deepseek_api_key on the command line.",
        }

    if not user_idea or not user_idea.strip():
        return {
            "success": False,
            "text": "",
            "message": "Please enter a topic or idea first.",
        }

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": user_idea.strip()},
        ],
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    print(f"[DeepSeek] Generating prompt for: {user_idea[:80]}...")

    try:
        resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        choices = data.get("choices", [])
        if choices:
            text = choices[0].get("message", {}).get("content", "").strip()
            if text:
                print(f"[DeepSeek] Generated {len(text)} chars")
                return {"success": True, "text": text, "message": "OK"}

        return {"success": False, "text": "",
                "message": f"Empty response from DeepSeek: {data}"}

    except requests.Timeout:
        return {"success": False, "text": "",
                "message": "DeepSeek API timed out. Try again."}
    except requests.RequestException as exc:
        return {"success": False, "text": "",
                "message": f"DeepSeek API error: {exc}"}
