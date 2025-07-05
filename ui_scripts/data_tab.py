from shiny import ui

def data_tab():
    return ui.nav_panel(
        "Data",
        ui.page_fluid(
            ui.div(
                ui.h3("Select Data Source"),
                ui.input_select(
                    "data_source",
                    "Select Data Source",
                    choices=["Upload", "Database", "API"]
                ),
                ui.input_file("file", "Upload Your File (.csv supported)", accept=[".csv"]),
                ui.div(
                    ui.download_button(
                        "download_template", 
                        "Download template file",
                        icon=ui.tags.i(class_="fas fa-download"),
                        class_="btn-info"
                    ),
                    ui.input_action_button(
                        "upload_data_btn", 
                        "Upload data",
                        icon=ui.tags.i(class_="fas fa-upload"),
                        class_="btn-primary"
                    ),
                    class_="d-flex justify-content-between my-3"
                ),
                ui.div(
                    class_="text-danger small",
                    content="Please click after editing (if needed) the table below"
                ),
                ui.div(
                    ui.h3("Edit Data"),
                    ui.output_data_frame("uploaded_data"),
                    class_="card p-3 mt-3"
                ),
                ui.div(
                    ui.h3("Quick Visualization"),
                    ui.output_plot("data_viz"),
                    class_="card p-3 mt-3"
                ),
                class_="card-body"
            )
        )
    )
