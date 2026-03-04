<div align="center">

# Offline AI Avatar Creator
### Generate Realistic Talking Faces

<p><strong>Upload an avatar video → Enter text → One click → Realistic talking-face video with cloned voice</strong></p>
<p>Powered by <a href="https://github.com/index-tts/index-tts">IndexTTS2</a> (Voice Cloning) + <a href="https://github.com/duixcom/Duix-Avatar">DuiX Avatar</a> (Video Synthesis)</p>

</div>

---
<img width="1885" height="910" alt="3" src="https://github.com/user-attachments/assets/acfa8ae3-8bf9-4fbd-a838-faa39f188194" />
<!-- ADD YOUR PLATFORM SCREENSHOT HERE -->
<!-- Example: ![Platform UI](assets/platform_screenshot.png) -->

## What is this?

This project combines two powerful open-source AI engines into a single offline platform:

- **[IndexTTS2](https://github.com/index-tts/index-tts)** — Zero-shot text-to-speech with voice cloning and emotion control by Bilibili
- **[DuiX Avatar](https://github.com/duixcom/Duix-Avatar)** — Talking-head face video synthesis with lip-sync by DuiX

Together they create an end-to-end pipeline: **text → cloned voice audio → realistic talking-face video**, all running locally on your GPU.

### ✨ Features

| Feature | Description |
|---|---|
| 🎤 **Voice Cloning** | Clone any voice from a short audio sample using IndexTTS2 zero-shot TTS |
| 📹 **Video Synthesis** | Generate realistic talking-face videos with lip-sync using DuiX Avatar engine |
| 🤖 **AI Script Generation** | DeepSeek AI writes speaking scripts from a brief idea or topic |
| 👤 **Avatar Library** | Save & reuse avatar + voice pairs — auto-saved after each generation |
| 😊 **Emotion Control** | 8 emotion sliders, reference audio, or text description to control voice tone |
| 🌐 **English & Chinese** | Full bilingual interface — switch between English and Chinese with one click |

### 📹 Example Output


https://github.com/user-attachments/assets/c73e22da-58f8-4560-b8e7-309e6f89d95c



<!-- ADD YOUR GENERATED VIDEO EXAMPLE HERE -->
<!-- Example: ![Example Video](assets/example_output.gif) -->
<!-- Or link to a video: [▶️ Watch example](https://your-video-link) -->

### 🖥️ How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Gradio UI  │────▶│  IndexTTS2   │────▶│  DuiX Avatar    │
│  (Web App)  │     │  (Voice      │     │  (Face Video    │
│             │     │   Cloning)   │     │   Synthesis)    │
└─────────────┘     └──────────────┘     └─────────────────┘
       │                                          │
       │            ┌──────────────┐              │
       └───────────▶│  DeepSeek AI │              │
                    │  (Script     │              │
                    │   Writer)    │              │
                    └──────────────┘              ▼
                                          🎬 Result Video
```

---

## 🚀 Installation

### Prerequisites

- Windows 10/11 or Ubuntu 22.04 with **NVIDIA GPU** (RTX 4070+ recommended)
- 32GB RAM, 100GB+ free disk space
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with GPU support
- [Git](https://git-scm.com/downloads) + [Git LFS](https://git-lfs.com/)
- [uv](https://docs.astral.sh/uv/) package manager

---

### Step 1: Install DuiX Avatar (Video Synthesis Engine)

DuiX Avatar provides the video synthesis backend. **You must install it first.**

👉 **Follow the full DuiX installation guide:** [duixcom/Duix-Avatar — How to Run Locally](https://github.com/duixcom/Duix-Avatar#3-how-to-run-locally)

Summary:

**1. Install Docker + NVIDIA drivers**

- **Windows:** Install [Docker Desktop](https://www.docker.com/), ensure WSL is installed (`wsl --list --verbose`), update with `wsl --update`
- **Ubuntu:** `sudo apt install docker.io docker-compose` + install [NVIDIA Container Toolkit](https://github.com/duixcom/Duix-Avatar#install-the-graphics-card-driver)
- Ensure your NVIDIA GPU driver is installed: [NVIDIA Driver Download](https://www.nvidia.cn/drivers/lookup/)

**2. Pull the Docker images:**

```bash
docker pull guiji2025/fun-asr
docker pull guiji2025/fish-speech-ziming
docker pull guiji2025/duix.avatar
```

**3. Clone DuiX and deploy:**

```bash
git clone https://github.com/duixcom/Duix-Avatar.git
cd Duix-Avatar/deploy
```

**Windows:**
```bash
docker-compose up -d
```

**Ubuntu:**
```bash
docker-compose -f docker-compose-linux.yml up -d
```

> **⚠️ First time download is ~70GB.** Takes about 30 minutes. Use WiFi or unlimited data.

**4. Verify** — You should see 3 containers running:

| Container | Port | Purpose |
|---|---|---|
| `duix-avatar-tts` | `18180` | Fish Speech TTS |
| `duix-avatar-asr` | `10095` | FunASR speech recognition |
| `duix-avatar-gen-video` | `8383` | Face2Face video synthesis |

---

### Step 2: Clone & Setup This Project

```bash
git clone https://github.com/jemi2k/ai-avatar-creator.git && cd ai-avatar-creator
git lfs pull
uv sync --all-extras
```

> [!TIP]
> Install `uv` quickly: `pip install -U uv`

---

### Step 3: Download IndexTTS2 Model (Voice Cloning Engine)

Via HuggingFace:
```bash
uv tool install "huggingface-hub[cli,hf_xet]"
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints
```

Or via ModelScope:
```bash
uv tool install "modelscope"
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

---

### Step 4: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your settings:
```env
DEEPSEEK_API_KEY=sk-your-actual-key-here
DUIX_API_URL=http://127.0.0.1:8384/easy
```

- **`DEEPSEEK_API_KEY`** — Get one at [platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys). Optional — only needed for the AI script generation feature.
- **`DUIX_API_URL`** — Default DuiX port is `8383`. If you changed it to `8384` due to a port conflict, update this value.

---

### Step 5: Launch

#### 5a. Start DuiX Docker Services First

Open a terminal and navigate to your DuiX Avatar deploy folder:

**Windows:**
```bash
cd C:\path\to\Duix-Avatar\deploy
docker-compose up -d
```

**Ubuntu:**
```bash
cd /path/to/Duix-Avatar/deploy
docker-compose -f docker-compose-linux.yml up -d
```

Verify all 3 containers are running:
```bash
docker ps
```

> **⚠️ Port conflict?** If port `8383` is already in use, edit the DuiX `docker-compose.yml` and change `'8383:8383'` to `'8384:8383'` for `duix-avatar-gen-video`. Then update your `.env`:
> ```env
> DUIX_API_URL=http://127.0.0.1:8384/easy
> ```

#### 5b. Start the AI Avatar Creator

Open a **new terminal** in this project folder:

**Windows:**
```bash
cd C:\path\to\index-tts
set HF_HUB_OFFLINE=1
uv run new_webui.py --port 7861
```

**Linux:**
```bash
cd /path/to/index-tts
export HF_HUB_OFFLINE=1
uv run new_webui.py --port 7861
```

Once you see `Running on local URL: http://127.0.0.1:7861`, open your browser at:

### 👉 http://127.0.0.1:7861

> [!IMPORTANT]
> **Always start DuiX Docker containers BEFORE launching this app.** Video synthesis will fail if DuiX services are not running.

---

## 📖 How to Use

1. **Upload** a source video (or select a saved avatar from the dropdown)
2. **Enter text** you want the avatar to speak (or click ✨ to generate a script with AI)
3. Click **🎬 Generate Video**
4. The avatar is automatically saved for future reuse

<!-- ADD YOUR USAGE SCREENSHOT/GIF HERE -->

---

## 📁 New Files Added

```
index-tts/
├── new_webui.py          # AI Avatar Creator WebUI
├── duix_integration.py   # DuiX API + DeepSeek + Avatar library
├── .env.example          # Environment config template
├── avatars/              # Saved avatar pairs (auto-created)
└── ...                   # All original IndexTTS2 files (unchanged)
```

## 🔧 CLI Options

```bash
uv run new_webui.py --help
```

| Flag | Default | Description |
|---|---|---|
| `--port` | `7860` | WebUI port |
| `--duix_url` | `http://127.0.0.1:8384/easy` | DuiX API endpoint |
| `--duix_data_root` | `D:\duix_avatar_data\face2face` | DuiX shared volume path |
| `--deepseek_api_key` | from `.env` | DeepSeek API key (overrides .env) |
| `--fp16` | off | Half-precision inference (less VRAM) |

---

## 🙏 Credits

- **[IndexTTS2](https://github.com/index-tts/index-tts)** by Bilibili IndexTeam — zero-shot voice cloning with emotion control
- **[DuiX Avatar](https://github.com/duixcom/Duix-Avatar)** by duixcom — talking-face video synthesis with lip-sync
- **[DeepSeek](https://platform.deepseek.com/)** — AI script generation API

## 📜 License

This project inherits the [Apache 2.0 License](LICENSE) from IndexTTS2.
DuiX Avatar is licensed under its own [license](https://github.com/duixcom/Duix-Avatar/blob/main/LICENSE).
