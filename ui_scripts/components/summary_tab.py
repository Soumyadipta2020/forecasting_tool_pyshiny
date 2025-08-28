from shiny import ui
from ui_scripts.components.common_ui import nav_panel, action_button, download_button


def summary_tab():
    return nav_panel(
        "Summary Statistics",
        ui.page_fluid(
            ui.br(),
            ui.card(
                ui.card_header(
                    "Summary Statistics"
                ),
                ui.card_body(
                    download_button(
                        "download_summary_stats",
                        "Download Summary Statistics",
                        icon_class="fas fa-download",
                    ),
                    ui.output_table("summary_stats")
                )
            ),
            ui.card(
                ui.card_header(
                    "Visualization",
                    ui.span(
                        action_button(
                            "viz_collapse_btn",
                            "",
                            icon_class="fas fa-minus",
                            class_="btn-sm btn-link text-light",
                        ),
                        class_="float-end"
                    )
                ),
                ui.card_body(
                    ui.div(
                        ui.h4("Plot Type"),
                        ui.input_select(
                            "plot_type",
                            "",
                            choices=[
                                "Boxplot",
                                "Violin Plot",
                                "Histogram"
                            ],
                            selected="Violin Plot"
                        ),
                        ui.output_plot("stats_viz")
                    )
                )
            )
        )
    )
