import os
import pytest

from test_config import *
from modules.live_portrait.live_portrait_inferencer import LivePortraitInferencer
from modules.utils.image_helper import save_image


@pytest.mark.parametrize(
    "input_image,expression_video",
    [
        (TEST_IMAGE_PATH, TEST_VIDEO_PATH),
    ]
)
def test_video_creation(
    input_image: str,
    expression_video: str
):
    if not os.path.exists(TEST_IMAGE_PATH):
        download_image(
            TEST_IMAGE_URL,
            TEST_IMAGE_PATH
        )
    if not os.path.exists(TEST_VIDEO_PATH):
        download_image(
            TEST_VIDEO_URL,
            TEST_VIDEO_PATH
        )

    inferencer = LivePortraitInferencer()

    output_video_path = inferencer.create_video(
        driving_vid_path=expression_video,
        src_image=input_image,
    )

    assert os.path.exists(output_video_path)
    assert validate_video(output_video_path)
    assert has_sound(output_video_path)
