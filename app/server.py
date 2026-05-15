"""
Server logic for the Data tab of the PyShiny AI Forecasting Application.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny import reactive, render, ui


APP_DIR = Path(__file__).resolve().parent
SAMPLE_FILE = APP_DIR / "timeseries_demo.csv"


def server_function(input, output, session):
    @reactive.calc
    @reactive.event(input.file, input.load_data, ignore_init=False)
    def loaded_data():
        if input.load_data() == 0 and not input.file():
            return None

        try:
            if input.data_source() == "Upload":
                file_info = input.file()
                if not file_info:
                    ui.notification_show("Please choose a CSV file before loading data.", type="error", duration=4)
                    return None
                return pd.read_csv(file_info[0]["datapath"])
            else:
                return pd.read_csv(SAMPLE_FILE)
        except Exception as exc:
            ui.notification_show(f"Could not load data: {exc}", type="error", duration=4)
            return None

    @reactive.effect
    def _sync_data_inputs():
        df = loaded_data()
        if df is None:
            return

        columns = df.columns.tolist()
        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        if not columns:
            return

        current_x = input.x_variables_graph()
        current_y = input.y_variable_graph()

        if current_x not in columns:
            current_x = columns[0]

        y_choices = [column for column in columns if column != current_x] or columns
        if current_y not in y_choices:
            current_y = next((column for column in numeric_columns if column in y_choices), y_choices[0])

        x_choices = [column for column in columns if column != current_y] or columns
        if current_x not in x_choices:
            current_x = x_choices[0]

        y_choices = [column for column in columns if column != current_x] or columns
        if current_y not in y_choices:
            current_y = next((column for column in numeric_columns if column in y_choices), y_choices[0])

        ui.update_select("time_variable", choices=columns, selected=columns[0], session=session)
        ui.update_select("y_variable_graph", choices=y_choices, selected=current_y, session=session)
        ui.update_select(
            "x_variables_graph",
            choices=x_choices,
            selected=current_x,
            session=session,
        )

    @output
    @render.ui
    def file_feedback():
        df = loaded_data()
        if df is None:
            return ui.div("Choose a data source, then load data.", class_="alert alert-info py-2")
        if df.empty:
            return ui.div("The loaded file is empty.", class_="alert alert-danger py-2")
        if not df.select_dtypes(include="number").columns.tolist():
            return ui.div("No numeric columns found. Please check the file format.", class_="alert alert-danger py-2")
        return ui.div("Data loaded successfully.", class_="alert alert-success py-2")

    @output
    @render.ui
    def info_data():
        df = loaded_data()
        if df is None:
            return ui.div(ui.p("Load data to see dataset statistics.", class_="text-muted"), class_="stat-grid")

        missing_pct = round(df.isna().sum().sum() / max(df.size, 1) * 100, 1)
        cards = [
            ("Observations", f"{df.shape[0]:,}", "tower-observation", "warning"),
            ("Variables", f"{df.shape[1]:,}", "square-root-variable", "success"),
            ("Missing %", f"{missing_pct}%", "percentage", "danger"),
            ("Numeric Columns", f"{len(df.select_dtypes(include='number').columns):,}", "list-ol", "info"),
        ]
        return ui.div(
            *(
                ui.div(
                    ui.div(ui.tags.i(class_=f"fa-solid fa-{icon}"), class_="stat-icon"),
                    ui.div(ui.div(value, class_="stat-value"), ui.div(label, class_="stat-label")),
                    class_=f"stat-card border-{color}",
                )
                for label, value, icon, color in cards
            ),
            class_="stat-grid",
        )

    @output
    @render.plot
    def vis_data():
        df = loaded_data()
        if df is None:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "Load data to see visualization", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return fig

        x_column = input.x_variables_graph()
        y_column = input.y_variable_graph()
        if not x_column or not y_column or x_column not in df.columns or y_column not in df.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "Select valid X and Y variables", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return fig

        y_values = pd.to_numeric(df[y_column], errors="coerce")
        if x_column == y_column:
            x_values = np.arange(1, len(df) + 1)
            x_label = "Sequence"
        else:
            x_values = df[x_column]
            parsed_dates = pd.to_datetime(x_values, errors="coerce")
            if parsed_dates.notna().all():
                x_values = parsed_dates
            x_label = x_column

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(x_values, y_values, marker="o", linewidth=1.8, color="#026efa", markersize=3)
        ax.set_title(f"{y_column} over {x_label}", fontsize=12)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_column)
        ax.grid(True, alpha=0.2)
        fig.autofmt_xdate()
        plt.tight_layout()
        return fig

    @output
    @render.download(filename="sample_data_template.csv")
    def file_template_download():
        yield pd.read_csv(SAMPLE_FILE).to_csv(index=False)
