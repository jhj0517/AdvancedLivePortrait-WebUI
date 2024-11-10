from PIL import Image, ImageChops
import requests
import os
import torch
import functools
import numpy as np
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from moviepy.editor import VideoFileClip

from modules.utils.paths import *


TEST_IMAGE_URL = "https://github.com/microsoft/onnxjs-demo/raw/master/src/assets/EmotionSampleImages/sad_baby.jpg"
TEST_VIDEO_URL = "https://github.com/jhj0517/sample-medias/raw/master/vids/human-face/expression01_short.mp4"
TEST_IMAGE_PATH = os.path.normpath(os.path.join(PROJECT_ROOT_DIR, "tests", "test.png"))
TEST_VIDEO_PATH = os.path.normpath(os.path.join(PROJECT_ROOT_DIR, "tests", "test_expression.mp4"))
TEST_EXPRESSION_OUTPUT_PATH = os.path.normpath(os.path.join(PROJECT_ROOT_DIR, "tests", "edited_expression.png"))
TEST_EXPRESSION_AAA = 100


def download_image(url, path):
    if os.path.exists(path):
       return

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Image successfully downloaded to {path}")
    else:
        raise Exception(f"Failed to download image. Status code: {response.status_code}")


def are_images_different(image1_path: str, image2_path: str):
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    diff = ImageChops.difference(image1, image2)

    if diff.getbbox() is None:
        return False
    else:
        return True


def are_videos_different(video1_path: str, video2_path: str):
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            if ret1 != ret2:
                return True
            break

        if frame1.shape != frame2.shape:
            frame1 = cv2.resize(frame1, (frame2.shape[1], frame2.shape[0]))

        score, _ = compare_ssim(frame1, frame2, full=True, multichannel=True)

        if score < 0.99:
            return True

    cap1.release()
    cap2.release()
    return False


def validate_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video file.")
        return False

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

    cap.release()

    if frame_count == 0:
        print("No frames found in video file.")
        return False

    return True


def has_sound(video_path: str):
    try:
        video = VideoFileClip(video_path)
        return video.audio is not None
    except Exception as e:
        return False


@functools.lru_cache
def is_cuda_available():
    return torch.cuda.is_available()

