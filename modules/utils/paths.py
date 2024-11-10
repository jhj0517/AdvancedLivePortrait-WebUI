import functools
import os


PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..")
MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, "models")
MODELS_ANIMAL_DIR = os.path.join(MODELS_DIR, "animal")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT_DIR, "outputs")
OUTPUTS_VIDEOS_DIR = os.path.join(OUTPUTS_DIR, "videos")
TEMP_DIR = os.path.join(OUTPUTS_DIR, "temp")
TEMP_VIDEO_FRAMES_DIR = os.path.join(TEMP_DIR, "video_frames")
TEMP_VIDEO_OUT_FRAMES_DIR = os.path.join(TEMP_VIDEO_FRAMES_DIR, "out")
EXP_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "exp_data")
MODEL_CONFIG = os.path.join(PROJECT_ROOT_DIR, "modules", "config", "models.yaml")
MODEL_PATHS = {
    "appearance_feature_extractor": os.path.join(MODELS_DIR, "appearance_feature_extractor.safetensors"),
    "motion_extractor": os.path.join(MODELS_DIR, "motion_extractor.safetensors"),
    "warping_module": os.path.join(MODELS_DIR, "warping_module.safetensors"),
    "spade_generator": os.path.join(MODELS_DIR, "spade_generator.safetensors"),
    "stitching_retargeting_module": os.path.join(MODELS_DIR, "stitching_retargeting_module.safetensors"),
    "face_yolov8n": os.path.join(MODELS_DIR, "face_yolov8n.pt")
}
MODEL_ANIMAL_PATHS = {
    "appearance_feature_extractor": os.path.join(MODELS_ANIMAL_DIR, "appearance_feature_extractor.safetensors"),
    "motion_extractor": os.path.join(MODELS_ANIMAL_DIR, "motion_extractor.safetensors"),
    "warping_module": os.path.join(MODELS_ANIMAL_DIR, "warping_module.safetensors"),
    "spade_generator": os.path.join(MODELS_ANIMAL_DIR, "spade_generator.safetensors"),
    "stitching_retargeting_module": os.path.join(MODELS_ANIMAL_DIR, "stitching_retargeting_module.safetensors"),
    # Just animal detection model not the face, needs better model
    "yolo_v5s_animal_det": os.path.join(MODELS_ANIMAL_DIR, "yolo_v5s_animal_det.n2x")
}
MASK_TEMPLATES = os.path.join(PROJECT_ROOT_DIR, "modules", "utils", "resources", "mask_template.png")
I18N_YAML_PATH = os.path.join(PROJECT_ROOT_DIR, "i18n", "translation.yaml")


def get_auto_incremental_file_path(dir_path: str, extension: str, prefix: str = ""):
    counter = len(os.listdir(dir_path))
    while True:
        if prefix:
            filename = f"{prefix}_{counter:05d}.{extension}"
        else:
            filename = f"{counter:05d}.{extension}"
        full_path = os.path.join(dir_path, filename)
        if not os.path.exists(full_path):
            full_path = os.path.normpath(full_path)
            return full_path
        counter += 1


@functools.lru_cache
def init_dirs():
    for dir_path in [
        MODELS_DIR,
        MODELS_ANIMAL_DIR,
        OUTPUTS_DIR,
        EXP_OUTPUT_DIR,
        TEMP_DIR,
        TEMP_VIDEO_FRAMES_DIR,
        TEMP_VIDEO_OUT_FRAMES_DIR,
        OUTPUTS_VIDEOS_DIR
    ]:
        os.makedirs(dir_path, exist_ok=True)


init_dirs()
