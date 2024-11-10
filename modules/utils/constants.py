from enum import Enum
from gradio_i18n import Translate, gettext as _


class ModelType(Enum):
    HUMAN = _("Human")
    ANIMAL = _("Animal")


class SamplePart(Enum):
    ONLY_EXPRESSION = _("OnlyExpression")
    ONLY_ROTATION = _("OnlyRotation")
    ONLY_MOUTH = _("OnlyMouth")
    ONLY_EYES = _("OnlyEyes")
    ALL = _("All")


REPO_MARKDOWN = """
## [AdvancedLivePortrait-WebUI](https://github.com/jhj0517/AdvancedLivePortrait-WebUI/tree/master)
"""

GRADIO_CSS = """
#md_project a {
  color: black;
  text-decoration: none;
}
#md_project a:hover {
  text-decoration: underline;
}

#blink_slider .md.svelte-7ddecg.chatbot.prose {
    font-size: 0.7em; 
}
"""

SOUND_FILE_EXT = ['.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a', '.wma']
IMAGE_FILE_EXT = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
VIDEO_FILE_EXT = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.mpeg', '.mpg', '.m4v', '.3gp', '.ts', '.vob', '.gif']
TRANSPARENT_VIDEO_FILE_EXT = ['.webm', '.mov', '.gif']
SUPPORTED_VIDEO_FILE_EXT = ['.mp4', '.mov', '.webm', '.gif']