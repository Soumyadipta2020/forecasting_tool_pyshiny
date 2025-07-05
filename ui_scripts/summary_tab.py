from shiny import ui

def summary_tab():
    return ui.nav_panel(
        "Summary Statistics",
        ui.page_fluid(
            ui.br(),
            ui.card(
                ui.card_header(
                    "Summary Statistics"
                ),
                ui.card_body(
                    ui.download_button(
                        "download_summary_stats",
                        "Download Summary Statistics",
                        icon=ui.tags.i(class_="fas fa-download")
                    ),
                    ui.output_table("summary_stats")
                )
            ),
            ui.card(
                ui.card_header(
                    "Visualization",
                    ui.span(
                        ui.input_action_button(
                            "viz_collapse_btn",
                            "",
                            icon=ui.tags.i(class_="fas fa-minus"),
                            class_="btn-sm btn-link text-light"
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
