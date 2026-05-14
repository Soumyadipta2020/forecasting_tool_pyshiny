from shiny import ui


def _icon(name: str):
    return ui.tags.i(class_=f"fa-solid fa-{name}")


def data_tab():
    return ui.nav_panel(
        "Data",
        ui.layout_sidebar(
            ui.sidebar(
                ui.card(
                    ui.card_header(_icon("database"), " Data Source"),
                    ui.input_select(
                        "data_source",
                        "Select Data Source",
                        choices=["Upload", "Sample"],
                        selected="Upload",
                    ),
                    ui.panel_conditional(
                        "input.data_source === 'Upload'",
                        ui.input_file(
                            "file",
                            "Upload Your File (.csv supported)",
                            accept=[".csv"],
                            multiple=False,
                        ),
                        ui.output_ui("file_feedback"),
                    ),
                    ui.panel_conditional(
                        "input.data_source === 'Sample'",
                        ui.input_select(
                            "sample_data",
                            "Select Sample Data",
                            choices=["timeseries demo"],
                            selected="timeseries demo",
                        ),
                        ui.input_action_button(
                            "load_mongo",
                            "Load Data",
                            icon=_icon("cloud-arrow-up"),
                            class_="btn-primary w-100",
                        ),
                    ),
                    ui.input_select("time_variable", "Select Time Variable", choices=[]),
                    ui.div(
                        ui.download_button(
                            "file_template_download",
                            "Download template file",
                            icon=_icon("download"),
                            class_="btn-info w-100",
                        ),
                        ui.input_action_button(
                            "upload_data",
                            "Upload data",
                            icon=_icon("upload"),
                            class_="btn-primary w-100",
                        ),
                        class_="button-column",
                    ),
                ),
                width=340,
            ),
            ui.div(
                ui.output_ui("info_data"),
                ui.card(
                    ui.card_header(_icon("chart-line"), " Quick Visualization"),
                    ui.layout_columns(
                        ui.input_select("y_variable_graph", "Select Y Variable", choices=[]),
                        ui.input_select("x_variables_graph", "Select X Variable", choices=[]),
                        col_widths=[6, 6],
                    ),
                    ui.output_plot("vis_data", height="420px"),
                    full_screen=True,
                ),
                class_="data-main-panel",
            ),
        ),
        value="data",
    )
