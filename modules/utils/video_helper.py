import subprocess
import os
from typing import List, Optional, Union
import cv2
from PIL import Image
import numpy as np
from dataclasses import dataclass
import re
from pathlib import Path

from modules.utils.constants import SOUND_FILE_EXT, VIDEO_FILE_EXT, IMAGE_FILE_EXT
from modules.utils.paths import (TEMP_VIDEO_FRAMES_DIR, TEMP_VIDEO_OUT_FRAMES_DIR, OUTPUTS_VIDEOS_DIR,
                                 get_auto_incremental_file_path)


@dataclass
class VideoInfo:
    num_frames: Optional[int] = None
    frame_rate: Optional[int] = None
    duration: Optional[float] = None
    has_sound: Optional[bool] = None
    codec: Optional[str] = None


def extract_frames(
    vid_input: str,
    output_temp_dir: str = TEMP_VIDEO_FRAMES_DIR,
    start_number: int = 0,
    clean=True
):
    """
    Extract frames as jpg files and save them into output_temp_dir. This needs FFmpeg installed.
    """
    if clean:
        clean_temp_dir(temp_dir=output_temp_dir)

    os.makedirs(output_temp_dir, exist_ok=True)
    output_path = os.path.join(output_temp_dir, "%05d.jpg")

    command = [
        'ffmpeg',
        '-loglevel', 'error',
        '-y',  # Enable overwriting
        '-i', vid_input,
        '-qscale:v', '2',
        '-vf', f'scale=iw:ih',
        '-start_number', str(start_number),
        f'{output_path}'
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Video frames extracted to \"{os.path.normpath(output_temp_dir)}\"")
    except subprocess.CalledProcessError as e:
        print("Error occurred while extracting frames from the video")
        raise RuntimeError(f"An error occurred: {str(e)}")

    return get_frames_from_dir(output_temp_dir)


def extract_sound(
    vid_input: str,
    output_temp_dir: str = TEMP_VIDEO_FRAMES_DIR,
):
    """
    Extract audio from a video file and save it as a separate sound file. This needs FFmpeg installed.
    """
    if Path(vid_input).suffix == ".gif":
        print("Sound extracting process has passed because gif has no sound")
        return None

    os.makedirs(output_temp_dir, exist_ok=True)
    output_path = os.path.join(output_temp_dir, "sound.mp3")

    command = [
        'ffmpeg',
        '-loglevel', 'error',
        '-y',  # Enable overwriting
        '-i', vid_input,
        '-vn',
        output_path
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to extract sound from the video: {e}")

    return output_path


def get_video_info(vid_input: str) -> VideoInfo:
    """
    Extract video information using ffmpeg.
    """
    command = [
        'ffmpeg',
        '-i', vid_input,
        '-map', '0:v:0',
        '-c', 'copy',
        '-f', 'null',
        '-'
    ]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                encoding='utf-8', errors='replace', check=True)
        output = result.stderr

        num_frames = None
        frame_rate = None
        duration = None
        has_sound = False
        codec = None

        for line in output.splitlines():
            if 'Stream #0:0' in line and 'Video:' in line:
                fps_match = re.search(r'(\d+(?:\.\d+)?) fps', line)
                if fps_match:
                    frame_rate = float(fps_match.group(1))

                codec_match = re.search(r'Video: (\w+)', line)
                if codec_match:
                    codec = codec_match.group(1)

            elif 'Duration:' in line:
                duration_match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})', line)
                if duration_match:
                    h, m, s = map(float, duration_match.groups())
                    duration = h * 3600 + m * 60 + s

            elif 'Stream' in line and 'Audio:' in line:
                has_sound = True

        if frame_rate and duration:
            num_frames = int(frame_rate * duration)

        print(f"Video info - frame_rate: {frame_rate}, duration: {duration}, total frames: {num_frames}")
        return VideoInfo(
            num_frames=num_frames,
            frame_rate=frame_rate,
            duration=duration,
            has_sound=has_sound,
            codec=codec
        )

    except subprocess.CalledProcessError as e:
        print("Error occurred while getting info from the video")
        return VideoInfo()


def create_video_from_frames(
    frames_dir: str,
    frame_rate: Optional[int] = None,
    sound_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    output_mime_type: Optional[str] = None,
):
    """
    Create a video from frames and save it to the output_path. This needs FFmpeg installed.
    """
    if not os.path.exists(frames_dir):
        raise "frames_dir does not exist"
    frames_dir = os.path.normpath(frames_dir)

    if output_dir is None:
        output_dir = OUTPUTS_VIDEOS_DIR
    os.makedirs(output_dir, exist_ok=True)

    frame_img_mime_type = ".png"
    pix_format = "yuv420p"
    vid_codec, audio_codec = "libx264", "aac"

    if output_mime_type is None:
        output_mime_type = ".mp4"

    output_mime_type = output_mime_type.lower()
    if output_mime_type == ".mov":
        pix_format = "yuva444p10le"
        vid_codec, audio_codec = "prores_ks", "aac"

    elif output_mime_type == ".webm":
        pix_format = "yuva420p"
        vid_codec, audio_codec = "libvpx-vp9", "libvorbis"

    elif output_mime_type == ".gif":
        pix_format = None
        vid_codec, audio_codec = "gif", None

    output_path = get_auto_incremental_file_path(output_dir, output_mime_type.replace(".", ""))

    if sound_path is None:
        temp_sound = os.path.normpath(os.path.join(TEMP_VIDEO_FRAMES_DIR, "sound.mp3"))
        if os.path.exists(temp_sound):
            sound_path = temp_sound

    if frame_rate is None:
        frame_rate = 25  # Default frame rate for ffmpeg

    command = [
        'ffmpeg',
        '-loglevel', 'error',
        '-y',
        '-framerate', str(frame_rate),
        '-i', os.path.join(frames_dir, f"%05d{frame_img_mime_type}"),
        '-c:v', vid_codec,
        '-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2' if pix_format else None,
    ]

    if output_mime_type == ".gif":
        command += [
            "-filter_complex", "[0:v] palettegen=reserve_transparent=on [p]; [0:v][p] paletteuse",
            "-loop", "0"
        ]
    else:
        command += [
            '-pix_fmt', pix_format
        ]

    command += [output_path]

    if output_mime_type != ".gif" and sound_path is not None:
        command += [
            '-i', sound_path,
            '-c:a', audio_codec,
            '-strict', 'experimental',
            '-b:a', '192k',
            '-shortest'
        ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while creating video from frames")
        raise
    return output_path


def create_video_from_numpy_list(frame_list: List[np.ndarray],
                                 frame_rate: Optional[int] = None,
                                 sound_path: Optional[str] = None,
                                 output_dir: Optional[str] = None
                                 ):
    if output_dir is None:
        output_dir = OUTPUTS_VIDEOS_DIR
    os.makedirs(output_dir, exist_ok=True)
    output_path = get_auto_incremental_file_path(output_dir, "mp4")

    if frame_rate is None:
        frame_rate = 25

    if sound_path is None:
        temp_sound = os.path.join(TEMP_VIDEO_FRAMES_DIR, "sound.mp3")
        if os.path.exists(temp_sound):
            sound_path = temp_sound

    height, width, layers = frame_list[0].shape
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    for frame in frame_list:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()


def get_frames_from_dir(vid_dir: str,
                        available_extensions: Optional[Union[List, str]] = None,
                        as_numpy: bool = False) -> List:
    """Get image file paths list from the dir"""
    if available_extensions is None:
        available_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG"]

    if isinstance(available_extensions, str):
        available_extensions = [available_extensions]

    frame_names = [
        p for p in os.listdir(vid_dir)
        if os.path.splitext(p)[-1] in available_extensions
    ]
    if not frame_names:
        return []
    frame_names.sort(key=lambda x: int(os.path.splitext(x)[0]))

    frames = [os.path.join(vid_dir, name) for name in frame_names]
    if as_numpy:
        frames = [np.array(Image.open(frame)) for frame in frames]

    return frames


def clean_temp_dir(temp_dir: Optional[str] = None):
    """Removes media files from the video frames directory."""
    if temp_dir is None:
        temp_dir = TEMP_VIDEO_FRAMES_DIR
        temp_out_dir = TEMP_VIDEO_OUT_FRAMES_DIR
    else:
        temp_out_dir = os.path.join(temp_dir, "out")

    clean_files_with_extension(temp_dir, SOUND_FILE_EXT)
    clean_files_with_extension(temp_dir, IMAGE_FILE_EXT)

    if os.path.exists(temp_out_dir):
        clean_files_with_extension(temp_out_dir, IMAGE_FILE_EXT)


def clean_files_with_extension(dir_path: str, extensions: List):
    """Remove files with the given extensions from the directory."""
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(tuple(extensions)):
            file_path = os.path.join(dir_path, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print("Error while removing image files")
