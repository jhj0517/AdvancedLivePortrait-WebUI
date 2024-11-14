import argparse
import gradio as gr
from gradio_i18n import Translate, gettext as _
from enum import Enum

from modules.live_portrait.live_portrait_inferencer import LivePortraitInferencer
from modules.utils.paths import *
from modules.utils.helper import str2bool
from modules.utils.constants import *


class App:
    def __init__(self,
                 args=None):
        self.args = args
        self.app = gr.Blocks(css=GRADIO_CSS)
        self.i18n = Translate(I18N_YAML_PATH)
        self.inferencer = LivePortraitInferencer(
            model_dir=args.model_dir if args else MODELS_DIR,
            output_dir=args.output_dir if args else OUTPUTS_DIR
        )
        self.reference_types = [src.value for src in ReferenceType]
        self.default_reference = self.reference_types[0]

    @staticmethod
    def create_expression_parameters():
        return [
            gr.Dropdown(label=_("Model Type"), visible=False, interactive=False,
                        choices=[item.value for item in ModelType], value=ModelType.HUMAN.value),
            gr.Slider(label=_("Rotate Pitch"), minimum=-20, maximum=20, step=0.5, value=0),
            gr.Slider(label=_("Rotate Yaw"), minimum=-20, maximum=20, step=0.5, value=0),
            gr.Slider(label=_("Rotate Roll"), minimum=-20, maximum=20, step=0.5, value=0),
            gr.Slider(label=_("Blink"), info=_("Value above 5 may appear distorted"), elem_id="blink_slider",
                      minimum=-20, maximum=20, step=0.5, value=0),
            gr.Slider(label=_("Eyebrow"), minimum=-40, maximum=20, step=0.5, value=0),
            gr.Slider(label=_("Wink"), minimum=0, maximum=25, step=0.5, value=0),
            gr.Slider(label=_("Pupil X"), minimum=-20, maximum=20, step=0.5, value=0),
            gr.Slider(label=_("Pupil Y"), minimum=-20, maximum=20, step=0.5, value=0),
            gr.Slider(label=_("AAA"), minimum=-30, maximum=120, step=1, value=0),
            gr.Slider(label=_("EEE"), minimum=-20, maximum=20, step=0.2, value=0),
            gr.Slider(label=_("WOO"), minimum=-20, maximum=20, step=0.2, value=0),
            gr.Slider(label=_("Smile"), minimum=-2.0, maximum=2.0, step=0.01, value=0),
            gr.Slider(label=_("Source Ratio"), minimum=0, maximum=1, step=0.01, value=1),
            gr.Slider(label=_("Sample Ratio"), minimum=-0.2, maximum=1.2, step=0.01, value=1, visible=False),
            gr.Dropdown(label=_("Sample Parts"), visible=False,
                        choices=[part.value for part in SamplePart], value=SamplePart.ALL.value),
            gr.Slider(label=_("Face Crop Factor"), minimum=1.5, maximum=2.5, step=0.1, value=2),
            gr.Checkbox(label=_("Enable Image Restoration"),
                        info=_("This enables image restoration with RealESRGAN but slows down the speed"), value=False)
        ]

    @staticmethod
    def create_video_parameters():
        return [
            gr.Dropdown(label=_("Model Type"), visible=False, interactive=False,
                        choices=[item.value for item in ModelType],
                        value=ModelType.HUMAN.value),
            gr.Slider(label=_("First frame eyes alignment factor"), minimum=0, maximum=1, step=0.01, value=1),
            gr.Slider(label=_("First frame mouth alignment factor"), minimum=0, maximum=1, step=0.01, value=1),
            gr.Slider(label=_("Face Crop Factor"), minimum=1.5, maximum=2.5, step=0.1, value=2),
            gr.Checkbox(label=_("Enable Image Restoration"),
                        info=_("This enables image restoration with RealESRGAN but slows down the speed"), value=False)
        ]

    @staticmethod
    def on_source_type_change_expression(source_type: str):
        return [
            gr.Image(label=_("Reference Image"), visible=source_type == ReferenceType.IMAGE.value),
            gr.Video(label=_("Reference Video"), visible=source_type != ReferenceType.IMAGE.value),
            gr.Button(_("GENERATE VIDEO"), visible=source_type != ReferenceType.IMAGE.value)
        ]

    @staticmethod
    def on_source_type_change_vid_driven(source_type: str):
        return [
            gr.Image(label=_("Reference Image"), visible=source_type == ReferenceType.IMAGE.value),
            gr.Video(label=_("Reference Video"), visible=source_type != ReferenceType.IMAGE.value),
        ]

    def launch(self):
        with self.app:
            with self.i18n:
                gr.Markdown(REPO_MARKDOWN, elem_id="md_project")

                with gr.Tabs():
                    with gr.TabItem(_("Expression Editor")):
                        with gr.Row():
                            with gr.Column():
                                img_ref = gr.Image(label=_("Reference Image"),
                                                   visible=self.default_reference == ReferenceType.IMAGE.value)
                                vid_ref = gr.Video(label=_("Reference Video"), interactive=True,
                                                   visible=self.default_reference != ReferenceType.IMAGE.value)
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=9):
                                img_out = gr.Image(label=_("Output Image"))
                                btn_gen = gr.Button(_("GENERATE VIDEO"), variant="primary", scale=0, min_width=5000,
                                                    visible=self.default_reference != ReferenceType.IMAGE.value)
                            with gr.Column(scale=1):
                                dd_src_type = gr.Dropdown(label=_("Reference Type"), choices=self.reference_types,
                                                          value=self.default_reference)
                                expression_parameters = self.create_expression_parameters()
                                btn_openfolder = gr.Button('📂')
                                with gr.Accordion("Opt in features", visible=False):
                                    img_sample = gr.Image()

                        params = [img_ref] + expression_parameters
                        opt_in_features_params = [img_sample]

                        gr.on(
                            triggers=[param.change for param in params],
                            fn=self.inferencer.edit_expression,
                            inputs=params + opt_in_features_params,
                            outputs=img_out,
                            show_progress="minimal",
                            queue=True
                        )

                        dd_src_type.change(
                            fn=self.on_source_type_change_expression,
                            inputs=[dd_src_type],
                            outputs=[img_ref, vid_ref, btn_gen]
                        )
                        btn_openfolder.click(
                            fn=lambda: self.open_folder(self.args.output_dir), inputs=None, outputs=None
                        )

                        btn_gen.click(self.inferencer.edit_expression,
                                      inputs=params + opt_in_features_params,
                                      outputs=img_out)

                    with gr.TabItem(_("Video Driven")):
                        with gr.Row():
                            img_ref = gr.Image(label=_("Reference Image"), interactive=True,
                                               visible=self.default_reference == ReferenceType.IMAGE.value)
                            vid_ref = gr.Video(label=_("Reference Video"), interactive=True,
                                               visible=self.default_reference != ReferenceType.IMAGE.value)
                            vid_driven = gr.Video(label=_("Expression Video"))
                            with gr.Column():
                                dd_src_type = gr.Dropdown(label=_("Reference Type"), choices=self.reference_types,
                                                          value=self.default_reference)
                                vid_params = self.create_video_parameters()

                        with gr.Row():
                            btn_gen = gr.Button(_("GENERATE VIDEO"), variant="primary")
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=9):
                                vid_out = gr.Video(label=_("Output Video"), scale=9)
                            with gr.Column(scale=1):
                                btn_openfolder = gr.Button('📂')

                        params = [img_ref, vid_driven] + vid_params

                        btn_gen.click(
                            fn=self.inferencer.create_video,
                            inputs=params,
                            outputs=vid_out
                        )

                        dd_src_type.change(
                            fn=self.on_source_type_change_vid_driven,
                            inputs=[dd_src_type],
                            outputs=[img_ref, vid_ref]
                        )
                        btn_openfolder.click(
                            fn=lambda: self.open_folder(os.path.join(self.args.output_dir, "videos")),
                            inputs=None, outputs=None
                        )

            gradio_launch_args = {
                "inbrowser": self.args.inbrowser,
                "share": self.args.share,
                "server_name": self.args.server_name,
                "server_port": self.args.server_port,
                "root_path": self.args.root_path,
                "auth": (self.args.username, self.args.password) if self.args.username and self.args.password else None,
            }
            self.app.queue(
                default_concurrency_limit=1
            ).launch(**gradio_launch_args)

    @staticmethod
    def open_folder(folder_path: str):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            print(f"The directory path {folder_path} has newly created.")
        os.system(f"start {folder_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--share', type=str2bool, default=False, nargs='?', const=True, help='Gradio share value')
    parser.add_argument('--inbrowser', type=str2bool, default=True, nargs='?', const=True,
                        help='Whether to automatically starts on the browser or not')
    parser.add_argument('--server_name', type=str, default=None, help='Gradio server host')
    parser.add_argument('--server_port', type=int, default=None, help='Gradio server port')
    parser.add_argument('--root_path', type=str, default=None, help='Gradio root path')
    parser.add_argument('--username', type=str, default=None, help='Gradio authentication username')
    parser.add_argument('--password', type=str, default=None, help='Gradio authentication password')
    parser.add_argument('--model_dir', type=str, default=MODELS_DIR,
                        help='Directory path of the LivePortrait models')
    parser.add_argument('--output_dir', type=str, default=OUTPUTS_DIR,
                        help='Directory path of the outputs')
    _args = parser.parse_args()

    app = App(args=_args)
    app.launch()




