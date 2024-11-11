import os
import pytest

from test_config import *
from modules.live_portrait.live_portrait_inferencer import LivePortraitInferencer


@pytest.mark.parametrize(
    "input_image",
    [
        TEST_IMAGE_PATH
    ]
)
def test_image_restoration(
    input_image: str,
):
    if not os.path.exists(TEST_IMAGE_PATH):
        download_image(
            TEST_IMAGE_URL,
            TEST_IMAGE_PATH
        )

    inferencer = LivePortraitInferencer()

    restored_output = inferencer.resrgan_inferencer.restore_image(
        input_image,
        overwrite=False
    )

    assert os.path.exists(restored_output)
    assert are_images_different(input_image, restored_output)
