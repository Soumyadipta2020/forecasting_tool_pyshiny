"""
Server logic for the Data tab of the PyShiny AI Forecasting Application.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import reactive, render, ui
from shinywidgets import render_widget


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

    def _normalize_selection(selection):
        if selection is None:
            return []
        if isinstance(selection, str):
            return [selection] if selection else []
        return [column for column in selection if column]

    def _summary_time_columns(df):
        time_columns = set()
        selected_time = input.time_variable()
        if selected_time in df.columns:
            time_columns.add(selected_time)
        time_columns.update(df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist())
        return time_columns

    def _summary_variable_choices(df):
        time_columns = _summary_time_columns(df)
        return [column for column in df.columns.tolist() if column not in time_columns]

    def _format_summary_value(value):
        if value is None or pd.isna(value):
            return ""
        if isinstance(value, (np.integer, int)):
            return f"{value:,}"
        if isinstance(value, (np.floating, float)):
            return f"{value:,.4f}".rstrip("0").rstrip(".")
        return str(value)

    def _series_mode(series):
        modes = series.dropna().mode()
        if modes.empty:
            return ""
        return modes.iloc[0]

    def _summary_statistics(df, selected_columns):
        rows = [
            "Data Type",
            "Observations",
            "Missing",
            "Unique",
            "Sum",
            "Mean",
            "Median",
            "Mode",
            "Standard Deviation",
            "Variance",
            "Minimum",
            "Maximum",
            "Skewness",
            "Kurtosis",
            "Interquartile Range",
        ]
        summary = pd.DataFrame({"Statistic": rows})

        for column in selected_columns:
            series = df[column]
            values = {
                "Data Type": str(series.dtype),
                "Observations": series.notna().sum(),
                "Missing": series.isna().sum(),
                "Unique": series.nunique(dropna=True),
                "Mode": _series_mode(series),
            }

            if pd.api.types.is_numeric_dtype(series):
                numeric_series = pd.to_numeric(series, errors="coerce")
                values.update(
                    {
                        "Sum": numeric_series.sum(skipna=True),
                        "Mean": numeric_series.mean(skipna=True),
                        "Median": numeric_series.median(skipna=True),
                        "Standard Deviation": numeric_series.std(skipna=True),
                        "Variance": numeric_series.var(skipna=True),
                        "Minimum": numeric_series.min(skipna=True),
                        "Maximum": numeric_series.max(skipna=True),
                        "Skewness": numeric_series.skew(skipna=True),
                        "Kurtosis": numeric_series.kurt(skipna=True),
                        "Interquartile Range": numeric_series.quantile(0.75) - numeric_series.quantile(0.25),
                    }
                )

            summary[column] = [_format_summary_value(values.get(row)) for row in rows]

        return summary

    def _empty_summary_plot(message):
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 16},
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(template="plotly_white", margin={"l": 24, "r": 24, "t": 48, "b": 24})
        return fig

    @reactive.effect
    @reactive.event(input.upload_data, ignore_init=True)
    def _():
        df = loaded_data()
        if df is None:
            ui.update_selectize("vars_stat_selected", choices=[], selected=[], session=session)
            return

        choices = _summary_variable_choices(df)
        ui.update_selectize("vars_stat_selected", choices=choices, selected=choices, session=session)

    @reactive.calc
    def summary_stat_data():
        df = loaded_data()
        if df is None:
            return pd.DataFrame({"Statistic": []})

        selected_columns = [
            column
            for column in _normalize_selection(input.vars_stat_selected())
            if column in _summary_variable_choices(df)
        ]
        if not selected_columns:
            return pd.DataFrame({"Statistic": []})

        return _summary_statistics(df, selected_columns)

    @output
    @render.ui
    def summary_status():
        df = loaded_data()
        if df is None:
            return ui.div("Click Upload data in the Data tab to prepare summary statistics.", class_="alert alert-info py-2")

        choices = _summary_variable_choices(df)
        if not choices:
            return ui.div("No non-time columns are available for summary statistics.", class_="alert alert-warning py-2")

        selected_columns = _normalize_selection(input.vars_stat_selected())
        if not selected_columns:
            return ui.div("Select one or more variables to see summary statistics.", class_="alert alert-info py-2")

        return ui.div("Summary statistics are ready.", class_="alert alert-success py-2")

    @output
    @render.data_frame
    def summary_stat_table():
        df = summary_stat_data()
        return render.DataGrid(
            df,
            width="100%",
            height="420px",
            filters=True,
            selection_mode="none",
        )

    @output
    @render.download(filename="summary_stat.csv")
    def summary_stat_download():
        yield summary_stat_data().to_csv(index=False)

    @output
    @render_widget
    def summary_stat_vis():
        df = loaded_data()
        if df is None:
            return _empty_summary_plot("Click Upload data to prepare summary statistics")

        selected_columns = [
            column
            for column in _normalize_selection(input.vars_stat_selected())
            if column in _summary_variable_choices(df)
        ]
        numeric_columns = [column for column in selected_columns if pd.api.types.is_numeric_dtype(df[column])]
        if not numeric_columns:
            return _empty_summary_plot("Select at least one numeric variable")

        plot_type = input.summary_stat_plot_type()
        fig = go.Figure()
        for column in numeric_columns:
            series = pd.to_numeric(df[column], errors="coerce").dropna()
            if plot_type == "Boxplot":
                fig.add_box(y=series, name=column, boxmean=True)
            elif plot_type == "Histogram":
                fig.add_histogram(x=series, name=column, opacity=0.6)
            else:
                fig.add_violin(y=series, name=column, box_visible=True, meanline_visible=True)

        fig.update_layout(
            title=plot_type,
            template="plotly_white",
            barmode="overlay" if plot_type == "Histogram" else None,
            margin={"l": 48, "r": 24, "t": 56, "b": 48},
        )
        return fig
