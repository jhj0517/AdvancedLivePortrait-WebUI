from modules.live_portrait.model_downloader import download_model

MODELS_REALESRGAN_URL = {
    "realesr-general-x4v3": "https://huggingface.co/jhj0517/realesr-general-x4v3/resolve/main/realesr-general-x4v3.pth",
    "RealESRGAN_x2": "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth",
}

MODELS_REALESRGAN_SCALABILITY = {
    "realesr-general-x4v3": [1, 2, 4],
    "RealESRGAN_x2": [2]
}


def download_resrgan_model(file_path, url):
    return download_model(file_path, url)
