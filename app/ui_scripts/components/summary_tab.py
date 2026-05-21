from shiny import ui
from shinywidgets import output_widget


def _icon(name: str):
    return ui.tags.i(class_=f"fa-solid fa-{name}")


def summary_tab():
    return ui.nav_panel(
        "Summary Statistics",
        ui.card(
            ui.card_header(_icon("chart-pie"), " Summary Statistics"),
            ui.output_ui("summary_status"),
            ui.input_selectize(
                "vars_stat_selected",
                "Select Variables",
                choices=[],
                multiple=True,
            ),
            ui.output_data_frame("summary_stat_table"),
            ui.download_button(
                "summary_stat_download",
                "Download Summary Statistics",
                icon=_icon("download"),
                class_="btn-info",
            ),
            full_screen=True,
        ),
        ui.card(
            ui.card_header(_icon("chart-simple"), " Visualization"),
            ui.input_select(
                "summary_stat_plot_type",
                "Plot Type",
                choices=["Boxplot", "Violin Plot", "Histogram"],
                selected="Violin Plot",
            ),
            output_widget("summary_stat_vis", height="380px", fill=False, fillable=False),
            class_="summary-visualization-card",
        ),
        ui.card(
            ui.card_header(_icon("arrow-right"), " Next Step"),
            ui.p("Review the summary statistics and visualization, then continue to forecasting."),
            ui.input_action_button(
                "implement_forecasting",
                "Implement Forecasting",
                icon=_icon("chart-line"),
                class_="btn-primary",
            ),
        ),
        value="summary",
    )
