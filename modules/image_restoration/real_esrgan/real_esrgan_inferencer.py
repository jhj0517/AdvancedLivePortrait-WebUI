import os.path
import gradio as gr
import torch
import cv2
from typing import Optional, Literal

from modules.utils.paths import *
from modules.utils.image_helper import save_image
from .model_downloader import download_resrgan_model, MODELS_REALESRGAN_URL, MODELS_REALESRGAN_SCALABILITY
from .wrapper.rrdb_net import RRDBNet
from .wrapper.real_esrganer import RealESRGANer
from .wrapper.srvgg_net_compact import SRVGGNetCompact


class RealESRGANInferencer:
    def __init__(self,
                 model_dir: str = MODELS_REAL_ESRGAN_DIR,
                 output_dir: str = OUTPUTS_DIR):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.device = self.get_device()
        self.arc = None
        self.model = None
        self.face_enhancer = None

        self.available_models = list(MODELS_REALESRGAN_URL.keys())
        self.default_model = self.available_models[0]
        self.model_config = {
            "model_name": self.default_model,
            "scale": 1,
            "half_precision": True
        }

    def load_model(self,
                   model_name: Optional[str] = None,
                   scale: Literal[1, 2, 4] = 1,
                   half_precision: bool = True,
                   progress: gr.Progress = gr.Progress()):
        model_config = {
            "model_name": model_name,
            "scale": scale,
            "half_precision": half_precision
        }
        if model_config == self.model_config and self.model is not None:
            return
        else:
            self.model_config = model_config

        if model_name is None:
            model_name = self.default_model

        model_path = os.path.join(self.model_dir, model_name)
        if not model_name.endswith(".pth"):
            model_path += ".pth"

        if not os.path.exists(model_path):
            progress(0, f"Downloading RealESRGAN model to : {model_path}")
            download_resrgan_model(model_path, MODELS_REALESRGAN_URL[model_name])

        name, ext = os.path.splitext(model_name)
        assert scale in MODELS_REALESRGAN_SCALABILITY[name]
        if name == 'RealESRGAN_x2':  # x4 RRDBNet model
            arc = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
        else:  # x4 VGG-style model (S size) : "realesr-general-x4v3"
            arc = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            netscale = 4

        self.model = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=arc,
            half=half_precision,
            device=torch.device(self.get_device())
        )

    def restore_image(self,
                      img_path: str,
                      model_name: Optional[str] = None,
                      scale: int = 1,
                      half_precision: Optional[bool] = None,
                      overwrite: bool = True):
        model_config = {
            "model_name": self.model_config["model_name"],
            "scale": scale,
            "half_precision": half_precision
        }
        half_precision = True if self.device == "cuda" else False

        if self.model is None or self.model_config != model_config:
            self.load_model(
                model_name=self.default_model if model_name is None else model_name,
                scale=scale,
                half_precision=half_precision
            )

        try:
            with torch.autocast(device_type=self.device, enabled=(self.device == "cuda")):
                output, img_mode = self.model.enhance(img_path, outscale=scale)
            if img_mode == "RGB":
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            if overwrite:
                output_path = img_path
            else:
                output_path = get_auto_incremental_file_path(self.output_dir, extension="png")

            output_path = save_image(output, output_path=output_path)
            return output_path
        except Exception as e:
            raise

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
