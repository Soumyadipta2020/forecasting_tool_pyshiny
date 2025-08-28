from shiny import ui
from ui_scripts.components.common_ui import nav_panel, file_input, download_button, action_button


def data_tab():
    return nav_panel(
        "Data",
        ui.page_fluid(
            ui.div(
                ui.h3("Select Data Source"),
                ui.input_select(
                    "data_source",
                    "Select Data Source",
                    choices=["Upload", "Database", "API"],
                ),
                file_input("file", "Upload Your File (.csv supported)", accept=[".csv"]),
                ui.div(
                    download_button(
                        "download_template",
                        "Download template file",
                        icon_class="fas fa-download",
                        class_="btn-info",
                    ),
                    action_button(
                        "upload_data_btn",
                        "Upload data",
                        icon_class="fas fa-upload",
                        class_="btn-primary",
                    ),
                    class_="d-flex justify-content-between my-3",
                ),
                ui.div(
                    class_="text-danger small",
                    content="Please click after editing (if needed) the table below",
                ),
                ui.div(
                    ui.h3("Edit Data"),
                    ui.output_data_frame("uploaded_data"),
                    class_="card p-3 mt-3",
                ),
                ui.div(
                    ui.h3("Quick Visualization"),
                    ui.output_plot("data_viz"),
                    class_="card p-3 mt-3",
                ),
                class_="card-body",
            )
        )
    )
