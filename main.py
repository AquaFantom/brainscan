import flet as ft
import os
import glob
from flet.core.types import ScrollMode
from ultralytics import YOLO
import torch


class BrainScan(ft.Column):
    def __init__(self, page: ft.Page):
        super().__init__()
        self.page = page
        self.page.scroll = ScrollMode.AUTO
        self.model = YOLO("best.pt").to("cuda" if torch.cuda.is_available() else "cpu")
        self.save_directory = "predictions"
        self.pick_files_dialog = ft.FilePicker(on_result=self.pick_files_result)
        self.selected_files_name = ft.Text()
        self.selected_files = None
        self.predicted_images = ft.Row(wrap=True)

        self.page.overlay.append(self.pick_files_dialog)
        self.predict_button = ft.FilledButton(text="Find tumors", on_click=self.process_image)

        self.page.add(ft.Column([
            ft.Row(
                [
                    ft.ElevatedButton(
                        "Pick files",
                        icon=ft.Icons.UPLOAD_FILE,
                        on_click=lambda _: self.pick_files_dialog.pick_files(
                            allow_multiple=True
                        ),
                    ),
                    self.selected_files_name,
                ]
            ),
            self.predict_button,
            self.predicted_images,
        ],)
        )
        self.empty_alert = ft.AlertDialog(
            modal=True,
            title=ft.Text("Alert"),
            content=ft.Text("Please select at least one file."),
            actions=[
                ft.TextButton("OK", on_click=lambda e: self.page.close(self.empty_alert)),
            ],
            actions_alignment=ft.MainAxisAlignment.END
        )

        self.page.update()


    def pick_files_result(self, e: ft.FilePickerResultEvent):
        self.selected_files_name.value = (
            ", ".join(map(lambda f: f.name, e.files)) if e.files else "Cancelled!"
        )
        self.selected_files = [file.path for file in e.files] if e.files else None
        self.selected_files_name.update()


    def process_image(self, e):
        if self.selected_files is None:
            self.page.open(self.empty_alert)
            return

        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        images = glob.glob(self.save_directory + "/*")
        for img in images:
            os.remove(img)
        self.predicted_images.controls = None
        self.predict_button.disabled = True
        self.page.update()

        predictions = self.model(self.selected_files)
        for pred, file_name in zip(predictions, self.selected_files_name.value.strip().split(", ")):
            pred.save(self.save_directory + "/pred_" + file_name)

        self.predict_button.disabled = False
        self.predict_button.update()

        images = glob.glob(self.save_directory + "/*")

        self.show_results(images)


    def show_results(self, images: list):
        for img in images:
            self.predicted_images.controls.append(ft.Image(src=img))
        self.page.update()



if __name__ == "__main__":
    def main(page: ft.Page):
        page.title = "BrainScan"
        page.padding = 25
        app = BrainScan(page)
        page.add(app)
        page.update()


    ft.app(main, assets_dir="assets")