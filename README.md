---
title: AdvancedLivePortrait-WebUI
emoji: 🚀
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.5.0
app_file: app.py
pinned: true
license: apache-2.0
---

# AdvancedLivePortrait-WebUI

Dedicated gradio based WebUI for [ComfyUI-AdvancedLivePortrait](https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait). 
<br>You can edit the facial expression from the image.

https://github.com/user-attachments/assets/cce42397-e898-4efb-9de8-735e050a10ce

## Notebook
You can try it in Colab 
- [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhj0517/AdvancedLivePortrait-WebUI/blob/master/notebooks/advanced_live_portrait_webui.ipynb)


# Installation And Running
### Prerequisite
1. `3.9` <= `python` <= `3.12` : https://www.python.org/downloads/release/python-3110/
2. **(Opitonal, only if you're using Nvidia GPU)** CUDA 12.4 : https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Windows
3. (Optional, only needed if you use Video Driven) `FFmpeg`:  https://ffmpeg.org/download.html <br> After installing `FFmpeg`, make sure to add the FFmpeg/bin folder to your **system PATH**!
## Run Locally
1. git clone this repository
```
git clone https://github.com/jhj0517/AdvancedLivePortrait-WebUI.git
```
2. Install dependencies ( Use `requirements-cpu.txt` if you're not using Nvidia GPU. )
```
pip install -r requirements.txt
```
3. Run app
```
python app.py
```

## Run with PowerShell
There're PowerShell scripts for each purpose : [`Install.ps1`](https://github.com/jhj0517/AdvancedLivePortrait-WebUI/blob/master/Install.ps1), [`Start-WebUI.ps1`](https://github.com/jhj0517/AdvancedLivePortrait-WebUI/blob/master/Start-WebUI.ps1) and [`Update.ps1`](https://github.com/jhj0517/AdvancedLivePortrait-WebUI/blob/master/Update.ps1).
<br> They do the same things as above with `venv`, creating, activating `venv` and running the app etc.

If you're using Windows, right-click the script and then click on ***Run with PowerShell***.

## Run with Docker
1. git clone this repository
```
git clone https://github.com/jhj0517/AdvancedLivePortrait-WebUI.git
```
2. Build the image
```
docker compose -f docker/docker-compose.yaml build
```
3. Run the container
```
docker compose -f docker/docker-compose.yaml up
```
4. Connect to `http://localhost:7860/` in browser.

Update the [`docker-compose.yaml`](https://github.com/jhj0517/AdvancedLivePortrait-WebUI/blob/master/docker/docker-compose.yaml) to match your environment if you're not using an Nvidia GPU.

## ❤️ Citation and Thanks
1. LivePortrait paper comes from
```bibtex
@article{guo2024liveportrait,
  title   = {LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control},
  author  = {Guo, Jianzhu and Zhang, Dingyun and Liu, Xiaoqiang and Zhong, Zhizhou and Zhang, Yuan and Wan, Pengfei and Zhang, Di},
  journal = {arXiv preprint arXiv:2407.03168},
  year    = {2024}
}
```
2. The models are safetensors that have been converted by kijai. : https://github.com/kijai/ComfyUI-LivePortraitKJ
3. [ultralytics](https://github.com/ultralytics/ultralytics) is used to detect the face.
4. This WebUI is started from [ComfyUI-AdvancedLivePortrait](https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait), various facial expressions like AAA, EEE, Eyebrow, Wink are found by PowerHouseMan.

### 🌐 Translation 
Any PRs for language translation for [`translation.yaml`](https://github.com/jhj0517/AdvancedLivePortrait-WebUI/blob/master/i18n/translation.yaml) would be greatly appreciated!


