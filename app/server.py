"""
Server logic for the Data tab of the PyShiny AI Forecasting Application.
"""

from pathlib import Path
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import reactive, render, ui
from shinywidgets import output_widget, render_widget


APP_DIR = Path(__file__).resolve().parent
SAMPLE_FILE = APP_DIR / "timeseries_demo.csv"


PLOT_BG = "#1f2b3d"
PLOT_GRID = "#334155"
PLOT_TEXT = "#e5edf8"
TIME_SERIES_MODEL_CHOICES = [
    "Naive",
    "Seasonal Naive",
    "Moving Average",
    "Drift",
    "ARIMA",
    "SARIMA",
    "ETS",
    "State Space ARIMA",
    "Prophet",
    "GRNN",
    "ARFIMA",
    "ARCH",
    "GARCH",
    "Neural Network",
    "AutoML",
    "Ensemble",
]
TABULAR_MODEL_CHOICES = [
    "Linear Regression",
    "GLM",
    "LASSO",
    "Ridge Regression",
    "Logistic Regression",
]


def _plotly_dark_layout(fig, title=None):
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font={"color": PLOT_TEXT},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.18,
            "yanchor": "top",
            "bgcolor": "rgba(31, 43, 61, 0)",
            "borderwidth": 0,
        },
        margin={"l": 48, "r": 24, "t": 48, "b": 76},
    )
    fig.update_xaxes(gridcolor=PLOT_GRID, zerolinecolor="#475569", title_standoff=14)
    fig.update_yaxes(gridcolor=PLOT_GRID, zerolinecolor="#475569", title_standoff=14)
    return fig


def _empty_plotly_figure(message):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15, "color": PLOT_TEXT},
    )
    _plotly_dark_layout(fig)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def server_function(input, output, session):
    data_cleanup_options = reactive.Value({"remove_duplicates": False})
    @reactive.calc
    def loaded_data():
        try:
            if input.data_source() == "Upload":
                file_info = input.file()
                if not file_info:
                    return None
                df = pd.read_csv(file_info[0]["datapath"])
            else:
                df = pd.read_csv(SAMPLE_FILE)

            cleanup = data_cleanup_options.get()
            if cleanup.get("remove_duplicates"):
                df = df.drop_duplicates().reset_index(drop=True)
            return df
        except Exception as exc:
            ui.notification_show(f"Could not load data: {exc}", type="error", duration=4)
            return None

    @reactive.calc
    def selected_time_column():
        return input.time_variable()

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

        selected_time = _likely_time_column(df) or columns[0]
        ui.update_select("time_variable", choices=columns, selected=selected_time, session=session)
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

    def _build_data_quality_summary(df, selected_time):
        rows = []
        recommendations = []

        total_missing = int(df.isna().sum().sum())
        missing_columns = int((df.isna().sum() > 0).sum())
        if total_missing:
            recommendations.append("Use interpolation or median/category imputation before modeling.")
        rows.append(("Missing values", total_missing, f"{missing_columns} affected columns"))

        duplicate_rows = int(df.duplicated().sum())
        if duplicate_rows:
            recommendations.append("Remove duplicate rows before training to avoid overweighting repeated observations.")
        rows.append(("Duplicate rows", duplicate_rows, "Exact row duplicates"))

        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        outlier_counts = {}
        for column in numeric_columns:
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            if len(values) < 4:
                continue
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0 or pd.isna(iqr):
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            count = int(((values < lower) | (values > upper)).sum())
            if count:
                outlier_counts[column] = count
        if outlier_counts:
            recommendations.append("Review high-outlier numeric columns; cap, smooth, or explain them before forecasting.")
        rows.append(("Outlier count", int(sum(outlier_counts.values())), ", ".join(f"{k}: {v}" for k, v in outlier_counts.items()) or "None"))

        non_numeric_issues = {}
        for column in df.columns:
            if column in numeric_columns:
                continue
            parsed = pd.to_numeric(df[column], errors="coerce")
            non_empty = df[column].notna()
            numeric_like = parsed.notna()
            if numeric_like.any():
                bad_count = int((non_empty & parsed.isna()).sum())
                if bad_count:
                    non_numeric_issues[column] = bad_count
        if non_numeric_issues:
            recommendations.append("Clean mixed numeric/text columns or explicitly treat them as categorical features.")
        rows.append(("Non-numeric issues", int(sum(non_numeric_issues.values())), ", ".join(f"{k}: {v}" for k, v in non_numeric_issues.items()) or "None"))

        date_gap_detail = "No date column selected"
        date_gap_count = 0
        if selected_time in df.columns:
            parsed_dates = pd.to_datetime(df[selected_time], errors="coerce").dropna().sort_values()
            if len(parsed_dates) >= 3:
                unique_dates = pd.Series(parsed_dates.unique()).sort_values()
                duplicate_dates = int(parsed_dates.duplicated().sum())
                inferred_freq = pd.infer_freq(unique_dates)
                if inferred_freq:
                    expected = pd.date_range(unique_dates.iloc[0], unique_dates.iloc[-1], freq=inferred_freq)
                    date_gap_count = max(0, len(expected) - len(unique_dates))
                    date_gap_detail = f"{date_gap_count} missing periods, {duplicate_dates} duplicate dates, inferred {inferred_freq}"
                else:
                    diffs = unique_dates.diff().dropna()
                    step = diffs.median() if not diffs.empty else pd.NaT
                    if pd.notna(step) and step != pd.Timedelta(0):
                        date_gap_count = int((diffs > step * 1.5).sum())
                        date_gap_detail = f"{date_gap_count} large gaps, {duplicate_dates} duplicate dates, median step {step}"
                    else:
                        date_gap_detail = f"Could not infer spacing, {duplicate_dates} duplicate dates"
                if date_gap_count or duplicate_dates:
                    recommendations.append("Fill missing time periods and remove duplicate timestamps before time-series forecasting.")
            else:
                date_gap_detail = "Not enough valid dates to assess gaps"
        rows.append(("Date gaps", date_gap_count, date_gap_detail))

        if not recommendations:
            recommendations.append("No major data quality issues detected. Proceed with validation before trusting forecasts.")

        return {
            "rows": rows,
            "recommendations": recommendations,
            "outliers": outlier_counts,
            "non_numeric": non_numeric_issues,
        }

    @reactive.calc
    def data_quality_summary():
        df = loaded_data()
        if df is None:
            return None
        return _build_data_quality_summary(df, selected_time_column())

    @output
    @render.ui
    def data_quality_report():
        quality = data_quality_summary()
        if quality is None:
            return ui.div("Load data to see missing values, duplicates, outliers, date gaps, and cleanup guidance.", class_="alert alert-info py-2")

        visual_meta = {
            "Duplicate rows": ("copy", "warning"),
            "Outlier count": ("triangle-exclamation", "purple"),
            "Non-numeric issues": ("font", "info"),
            "Date gaps": ("calendar-xmark", "success"),
        }
        visible_rows = [row for row in quality["rows"] if row[0] != "Missing values"]
        compact_detail = {
            "Duplicate rows": "Duplicates",
            "Outlier count": "IQR flags",
            "Non-numeric issues": "Mixed values",
            "Date gaps": "Timeline gaps",
        }
        issue_count = sum(count for _, count, _ in visible_rows)
        status_text = "Clean" if issue_count == 0 else f"{issue_count:,} issues"
        status_class = "quality-status is-clean" if issue_count == 0 else "quality-status"
        return ui.div(
            ui.div(
                *(
                    ui.div(
                        ui.div(ui.tags.i(class_=f"fa-solid fa-{visual_meta.get(issue, ('clipboard-check', 'primary'))[0]}"), class_="quality-icon"),
                        ui.div(
                            ui.div(issue, class_="quality-label"),
                            ui.div(f"{count:,}", class_="quality-value"),
                            ui.div(compact_detail.get(issue, detail), class_="quality-detail"),
                        ),
                        class_=f"quality-card border-{visual_meta.get(issue, ('clipboard-check', 'primary'))[1]}",
                    )
                    for issue, count, detail in visible_rows
                ),
                class_="quality-grid",
            ),
            ui.div(
                ui.div(ui.tags.i(class_="fa-solid fa-circle-check"), status_text, class_=status_class),
                ui.div(
                    quality["recommendations"][0] if quality["recommendations"] else "Ready for validation.",
                    class_="quality-status-copy",
                ),
                class_="quality-recommendations compact",
            ),
            ui.div(
                ui.input_action_button(
                    "apply_missing_imputation",
                    "Use interpolation",
                    icon=ui.tags.i(class_="fa-solid fa-wand-magic-sparkles"),
                    class_="btn-info",
                    disabled=quality["rows"][0][1] == 0,
                ),
                ui.input_action_button(
                    "apply_outlier_capping",
                    "Cap outliers",
                    icon=ui.tags.i(class_="fa-solid fa-scissors"),
                    class_="btn-info",
                    disabled=not bool(quality["outliers"]),
                ),
                ui.input_action_button(
                    "remove_duplicate_rows",
                    "Remove duplicates",
                    icon=ui.tags.i(class_="fa-solid fa-copy"),
                    class_="btn-secondary",
                    disabled=not any(issue == "Duplicate rows" and count for issue, count, _ in quality["rows"]),
                ),
                class_="quality-action-row",
            ),
        )

    @reactive.effect
    @reactive.event(input.apply_missing_imputation, ignore_init=True)
    def _apply_missing_imputation():
        ui.update_checkbox("preprocess_interpolate", value=True, session=session)
        ui.notification_show("Interpolation enabled in Forecast advanced settings.", type="message", duration=4)

    @reactive.effect
    @reactive.event(input.apply_outlier_capping, ignore_init=True)
    def _apply_outlier_capping():
        ui.update_checkbox("preprocess_outliers", value=True, session=session)
        ui.notification_show("Outlier capping enabled in Forecast advanced settings.", type="message", duration=4)

    @reactive.effect
    @reactive.event(input.remove_duplicate_rows, ignore_init=True)
    def _remove_duplicate_rows():
        options = data_cleanup_options.get().copy()
        options["remove_duplicates"] = True
        data_cleanup_options.set(options)
        ui.notification_show("Duplicate rows removed from the active dataset.", type="message", duration=4)

    @output
    @render_widget
    def vis_data():
        df = loaded_data()
        if df is None:
            return _empty_plotly_figure("Load data to see visualization")

        x_column = input.x_variables_graph()
        y_column = input.y_variable_graph()
        if not x_column or not y_column or x_column not in df.columns or y_column not in df.columns:
            return _empty_plotly_figure("Select valid X and Y variables")

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

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines+markers",
                name=y_column,
                line={"color": "#3487ff", "width": 2.2},
                marker={"color": "#06b6f0", "size": 6},
            )
        )
        _plotly_dark_layout(fig, title=f"{y_column} over {x_label}")
        fig.update_xaxes(title=x_label)
        fig.update_yaxes(title=y_column)
        return fig

    @output
    @render.download(filename="sample_data_template.csv")
    def file_template_download():
        yield pd.read_csv(SAMPLE_FILE).to_csv(index=False)

    @output
    @render.ui
    def data_action_controls():
        df = loaded_data()
        ready = df is not None and not df.empty and bool(df.select_dtypes(include="number").columns.tolist())
        return ui.div(
            ui.download_button(
                "file_template_download",
                "Download CSV template",
                icon=ui.tags.i(class_="fa-solid fa-download"),
                class_="btn-info w-100",
            ),
            ui.input_action_button(
                "upload_data",
                "Continue to summary",
                icon=ui.tags.i(class_="fa-solid fa-arrow-right"),
                class_="btn-primary w-100",
                disabled=not ready,
            ),
            class_="button-column",
        )

    @reactive.effect
    @reactive.event(input.upload_data, ignore_init=True)
    def _navigate_to_summary():
        df = loaded_data()
        if df is None or df.empty:
            ui.notification_show("Load a valid dataset before continuing.", type="warning", duration=4)
            return
        ui.update_navset("main_nav", selected="summary", session=session)

    def _normalize_selection(selection):
        if selection is None:
            return []
        if isinstance(selection, str):
            return [selection] if selection else []
        return [column for column in selection if column]

    def _build_summary_time_columns(df, selected_time):
        time_columns = set()
        if selected_time in df.columns:
            time_columns.add(selected_time)
        time_columns.update(df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist())
        return time_columns

    def _build_summary_variable_choices(df, selected_time):
        time_columns = _build_summary_time_columns(df, selected_time)
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
        _plotly_dark_layout(fig)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig

    @reactive.calc
    def summary_variable_choices():
        df = loaded_data()
        if df is None:
            return []
        return _build_summary_variable_choices(df, selected_time_column())

    @reactive.calc
    def selected_summary_columns():
        choices = summary_variable_choices()
        return [
            column
            for column in _normalize_selection(input.vars_stat_selected())
            if column in choices
        ]

    @reactive.effect
    @reactive.event(input.upload_data, ignore_init=True)
    def _populate_summary_controls():
        df = loaded_data()
        if df is None:
            ui.update_selectize("vars_stat_selected", choices=[], selected=[], session=session)
            return

        choices = summary_variable_choices()
        ui.update_selectize("vars_stat_selected", choices=choices, selected=choices, session=session)

    @reactive.calc
    def summary_stat_data():
        df = loaded_data()
        if df is None:
            return pd.DataFrame({"Statistic": []})

        selected_columns = selected_summary_columns()
        if not selected_columns:
            return pd.DataFrame({"Statistic": []})

        return _summary_statistics(df, selected_columns)

    @output
    @render.ui
    def summary_status():
        df = loaded_data()
        if df is None:
            return ui.div("Click Upload data in the Data tab to prepare summary statistics.", class_="alert alert-info py-2")

        choices = summary_variable_choices()
        if not choices:
            return ui.div("No non-time columns are available for summary statistics.", class_="alert alert-warning py-2")

        selected_columns = selected_summary_columns()
        if not selected_columns:
            return ui.div("Select one or more variables to see summary statistics.", class_="alert alert-info py-2")

        return ui.div("Summary statistics are ready.", class_="alert alert-success py-2")

    @output
    @render.ui
    def summary_next_actions():
        df = loaded_data()
        ready = df is not None and not df.empty and bool(df.select_dtypes(include="number").columns.tolist())
        return ui.div(
            ui.p("Review the selected variables, then continue to forecasting.", class_="next-step-copy"),
            ui.input_action_button(
                "implement_forecasting",
                "Continue to forecasting",
                icon=ui.tags.i(class_="fa-solid fa-chart-line"),
                class_="btn-primary",
                disabled=not ready,
            ),
            class_="summary-next-actions",
        )

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

        selected_columns = selected_summary_columns()
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

        _plotly_dark_layout(fig, title=plot_type)
        fig.update_layout(barmode="overlay" if plot_type == "Histogram" else None)
        return fig

    forecast_result = reactive.Value(None)

    def _as_tuple(value):
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        try:
            return tuple(value)
        except TypeError:
            return (value,)

    @reactive.calc
    def _forecast_signature():
        return (
            input.data_source(),
            input.time_variable(),
            input.data_type(),
            input.response_variable(),
            _valid_horizon(),
            _as_tuple(input.model()),
            _as_tuple(input.model1()),
            _as_tuple(input.x_variables()),
            bool(input.preprocess_interpolate()),
            bool(input.preprocess_outliers()),
            float(input.scenario_adj() or 0),
            bool(input.seasonal()),
            int(input.seasonal_period() or 0),
            input.ts_validation(),
            int(input.ts_test_periods() or 0),
            int(input.rolling_folds() or 0),
            int(input.rolling_initial_pct() or 0),
            input.best_model_metric(),
            bool(input.use_backtesting()),
            input.tabular_split_mode(),
            int(input.train_split() or 0),
            int(input.tabular_test_rows() or 0),
            bool(input.use_scenario()),
            input.scenario_feature(),
            input.scenario_value(),
        )

    def _forecast_is_stale(result):
        return bool(result and result.get("ok") and result.get("signature") != _forecast_signature())

    def _numeric_columns(df):
        return df.select_dtypes(include="number").columns.tolist()

    def _likely_time_column(df):
        if df is None or df.empty:
            return None
        name_tokens = ("date", "time", "year", "month", "period", "week", "day")
        columns = df.columns.tolist()
        for column in columns:
            if any(token in column.lower() for token in name_tokens):
                return column
        for column in columns:
            parsed = pd.to_datetime(df[column], errors="coerce")
            if parsed.notna().sum() >= max(3, int(len(df) * 0.75)):
                return column
        return columns[0] if columns else None

    def _likely_response_column(df):
        numeric_columns = _numeric_columns(df)
        if not numeric_columns:
            return None
        name_tokens = ("sales", "price", "revenue", "demand", "forecast", "value", "amount", "quantity", "volume")
        for column in numeric_columns:
            if any(token in column.lower() for token in name_tokens):
                return column
        return numeric_columns[0]

    @reactive.calc
    def _valid_horizon():
        try:
            return max(1, int(input.horizon() or 1))
        except (TypeError, ValueError):
            return 1

    def _preferred_response_column(df):
        numeric_columns = _numeric_columns(df)
        if not numeric_columns:
            return None

        current_response = input.response_variable()
        if current_response in numeric_columns:
            return current_response

        current_graph_y = input.y_variable_graph()
        if current_graph_y in numeric_columns:
            return current_graph_y

        return _likely_response_column(df) or numeric_columns[0]

    def _feature_columns(df, response_column):
        selected_columns = [
            column
            for column in _normalize_selection(input.x_variables())
            if column in df.columns and column != response_column
        ]
        if selected_columns:
            return selected_columns

        return [
            column
            for column in df.columns
            if column != response_column
        ]

    @reactive.calc
    def selected_response_column():
        df = loaded_data()
        if df is None:
            return None
        return _preferred_response_column(df)

    @reactive.calc
    def selected_feature_columns():
        df = loaded_data()
        response_column = input.response_variable()
        if df is None or response_column not in df.columns:
            return []
        return _feature_columns(df, response_column)

    def _update_forecast_controls(df, response_column=None, select_all_x=False):
        numeric_columns = _numeric_columns(df)
        selected_response = response_column or _preferred_response_column(df)

        ui.update_select(
            "response_variable",
            choices=numeric_columns,
            selected=selected_response,
            session=session,
        )

        x_choices = [
            column
            for column in df.columns.tolist()
            if column != selected_response
        ]
        current_x = [
            column
            for column in _normalize_selection(input.x_variables())
            if column in x_choices
        ]
        ui.update_selectize(
            "x_variables",
            choices=x_choices,
            selected=x_choices if select_all_x else current_x or x_choices,
            session=session,
        )
        ui.update_select(
            "scenario_feature",
            choices=x_choices,
            selected=x_choices[0] if x_choices else None,
            session=session,
        )

    @reactive.effect
    def _sync_forecast_inputs():
        df = loaded_data()
        if df is None:
            ui.update_select("response_variable", choices=[], selected=None, session=session)
            ui.update_selectize("x_variables", choices=[], selected=[], session=session)
            return

        with reactive.isolate():
            _update_forecast_controls(df, select_all_x=True)

    @reactive.effect
    @reactive.event(input.response_variable, ignore_init=True)
    def _sync_x_variables_to_response():
        df = loaded_data()
        if df is None:
            return

        with reactive.isolate():
            response_column = input.response_variable()
            if response_column in df.columns:
                _update_forecast_controls(df, response_column=response_column, select_all_x=True)

    def _as_error(message):
        return {"ok": False, "message": message}

    def _regression_metrics(actual, predicted, k=2, training_series=None, lower=None, upper=None):
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        mask = np.isfinite(actual) & np.isfinite(predicted)
        if not mask.any():
            return {
                "MAE": np.nan,
                "RMSE": np.nan,
                "MAPE": np.nan,
                "sMAPE": np.nan,
                "MASE": np.nan,
                "WAPE": np.nan,
                "MdAPE": np.nan,
                "R2": np.nan,
                "BIC": np.nan,
                "Coverage": np.nan,
                "Avg Width": np.nan,
            }

        actual = actual[mask]
        predicted = predicted[mask]
        errors = actual - predicted
        mae = float(np.mean(np.abs(errors)))
        ssr = float(np.sum(errors**2))
        n = len(actual)
        rmse = float(np.sqrt(ssr / n))
        nonzero = actual != 0
        mape = float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100) if nonzero.any() else np.nan
        mdape = float(np.median(np.abs(errors[nonzero] / actual[nonzero])) * 100) if nonzero.any() else np.nan
        smape_denominator = np.abs(actual) + np.abs(predicted)
        smape_mask = smape_denominator != 0
        smape = float(np.mean(2 * np.abs(errors[smape_mask]) / smape_denominator[smape_mask]) * 100) if smape_mask.any() else np.nan
        actual_sum = float(np.sum(np.abs(actual)))
        wape = float(np.sum(np.abs(errors)) / actual_sum * 100) if actual_sum else np.nan

        scale_source = np.asarray(training_series if training_series is not None else actual, dtype=float)
        scale_source = scale_source[np.isfinite(scale_source)]
        naive_scale = float(np.mean(np.abs(np.diff(scale_source)))) if len(scale_source) > 1 else np.nan
        mase = float(mae / naive_scale) if np.isfinite(naive_scale) and naive_scale != 0 else np.nan

        total = float(np.sum((actual - np.mean(actual)) ** 2))
        r2 = float(1 - ssr / total) if total else np.nan
        
        # Calculate pseudo-BIC
        if ssr > 0 and n > 0:
            bic = float(n * np.log(ssr / n) + k * np.log(n))
        else:
            bic = np.nan

        coverage = np.nan
        avg_width = np.nan
        if lower is not None and upper is not None:
            lower = np.asarray(lower, dtype=float)
            upper = np.asarray(upper, dtype=float)
            interval_mask = mask.copy()
            if len(lower) == len(mask) and len(upper) == len(mask):
                lower = lower[interval_mask]
                upper = upper[interval_mask]
            elif len(lower) != len(actual) or len(upper) != len(actual):
                lower = upper = np.asarray([], dtype=float)
            if len(lower) == len(actual) and len(upper) == len(actual):
                interval_valid = np.isfinite(lower) & np.isfinite(upper)
                if interval_valid.any():
                    coverage = float(np.mean((actual[interval_valid] >= lower[interval_valid]) & (actual[interval_valid] <= upper[interval_valid])) * 100)
                    avg_width = float(np.mean(upper[interval_valid] - lower[interval_valid]))

        return {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "sMAPE": smape,
            "MASE": mase,
            "WAPE": wape,
            "MdAPE": mdape,
            "R2": r2,
            "BIC": bic,
            "Coverage": coverage,
            "Avg Width": avg_width,
        }

    def _classification_metrics(actual, predicted):
        actual = np.asarray(actual)
        predicted = np.asarray(predicted)
        mask = pd.notna(actual) & pd.notna(predicted)
        if not mask.any():
            return {"Accuracy": np.nan}
        return {"Accuracy": float(np.mean(actual[mask] == predicted[mask]) * 100)}

    def _format_metric(value, suffix=""):
        if value is None or pd.isna(value):
            return "N/A"
        return f"{value:,.2f}{suffix}"

    def _future_axis_from_time(df, horizon):
        time_column = selected_time_column()
        if time_column not in df.columns:
            actual_axis = np.arange(1, len(df) + 1)
            return actual_axis, np.arange(len(df) + 1, len(df) + horizon + 1), "Sequence"

        raw_axis = df[time_column].reset_index(drop=True)
        numeric_axis = pd.to_numeric(raw_axis, errors="coerce")
        if pd.api.types.is_numeric_dtype(raw_axis) or numeric_axis.notna().sum() >= max(2, int(len(numeric_axis) * 0.8)):
            valid_axis = numeric_axis.dropna()
            step = valid_axis.diff().dropna().median()
            if pd.isna(step) or step == 0:
                step = 1
            future_axis = np.asarray([valid_axis.iloc[-1] + step * (i + 1) for i in range(horizon)])
            return numeric_axis, future_axis, time_column

        parsed_dates = pd.to_datetime(raw_axis, errors="coerce")
        if parsed_dates.notna().sum() >= max(2, int(len(parsed_dates) * 0.8)):
            actual_axis = parsed_dates
            valid_dates = parsed_dates.dropna()
            frequency = pd.infer_freq(valid_dates)
            if frequency:
                future_axis = pd.date_range(valid_dates.iloc[-1], periods=horizon + 1, freq=frequency)[1:]
            elif len(valid_dates) >= 2:
                step = valid_dates.diff().dropna().median()
                if pd.isna(step) or step == pd.Timedelta(0):
                    step = pd.Timedelta(days=1)
                future_axis = pd.date_range(valid_dates.iloc[-1] + step, periods=horizon, freq=step)
            else:
                future_axis = pd.date_range(valid_dates.iloc[-1] + pd.Timedelta(days=1), periods=horizon)
            return actual_axis, future_axis, time_column

        actual_axis = np.arange(1, len(df) + 1)
        return actual_axis, np.arange(len(df) + 1, len(df) + horizon + 1), "Sequence"

    def _time_series_profile(df, response_column=None):
        time_column = selected_time_column()
        profile = {
            "time_column": time_column if time_column in df.columns else None,
            "frequency": None,
            "frequency_label": "sequence",
            "detected_period": None,
            "trend": "flat",
            "date_gaps": 0,
            "duplicate_timestamps": 0,
            "notes": [],
        }

        if response_column in df.columns:
            y = pd.to_numeric(df[response_column], errors="coerce").dropna().astype(float)
            if len(y) >= 3:
                slope = np.polyfit(np.arange(len(y)), y.to_numpy(), 1)[0]
                denom = np.nanstd(y.to_numpy()) or 1.0
                profile["trend"] = "upward" if slope > denom * 0.02 else "downward" if slope < -denom * 0.02 else "flat"
                profile["detected_period"] = _detect_seasonal_period(y)

        if time_column not in df.columns:
            return profile

        parsed_dates = pd.to_datetime(df[time_column], errors="coerce").dropna().sort_values()
        if len(parsed_dates) < 3:
            return profile

        unique_dates = pd.Series(parsed_dates.unique()).sort_values()
        profile["duplicate_timestamps"] = int(parsed_dates.duplicated().sum())
        inferred = pd.infer_freq(unique_dates)
        profile["frequency"] = inferred
        if inferred:
            profile["frequency_label"] = inferred
            try:
                expected = pd.date_range(unique_dates.iloc[0], unique_dates.iloc[-1], freq=inferred)
                profile["date_gaps"] = max(0, len(expected) - len(unique_dates))
            except Exception:
                profile["date_gaps"] = 0
        else:
            diffs = unique_dates.diff().dropna()
            median_step = diffs.median() if not diffs.empty else pd.NaT
            if pd.notna(median_step) and median_step != pd.Timedelta(0):
                profile["date_gaps"] = int((diffs > median_step * 1.5).sum())
                if median_step <= pd.Timedelta(hours=1):
                    profile["frequency_label"] = "sub-daily"
                elif median_step <= pd.Timedelta(days=1):
                    profile["frequency_label"] = "daily-ish"
                elif median_step <= pd.Timedelta(days=8):
                    profile["frequency_label"] = "weekly-ish"
                elif median_step <= pd.Timedelta(days=32):
                    profile["frequency_label"] = "monthly-ish"

        period = profile.get("detected_period")
        if not period:
            freq_label = str(profile["frequency_label"]).upper()
            if freq_label.startswith("D") or "DAILY" in freq_label:
                profile["detected_period"] = 7
            elif freq_label.startswith("W") or "WEEKLY" in freq_label:
                profile["detected_period"] = 52
            elif freq_label.startswith("M") or "MONTH" in freq_label:
                profile["detected_period"] = 12
            elif freq_label.startswith("Q"):
                profile["detected_period"] = 4
            elif freq_label.startswith("H") or "SUB-DAILY" in freq_label:
                profile["detected_period"] = 24

        if profile["date_gaps"]:
            profile["notes"].append(f"{profile['date_gaps']} possible missing time periods")
        if profile["duplicate_timestamps"]:
            profile["notes"].append(f"{profile['duplicate_timestamps']} duplicate timestamps")
        return profile

    def _conformal_interval(future, y, fitted=None, validation_actual=None, validation_predicted=None):
        future = np.asarray(future, dtype=float)
        residuals = np.asarray([], dtype=float)
        if validation_actual is not None and validation_predicted is not None:
            validation_actual = np.asarray(validation_actual, dtype=float)
            validation_predicted = np.asarray(validation_predicted, dtype=float)
            mask = np.isfinite(validation_actual) & np.isfinite(validation_predicted)
            residuals = np.abs(validation_actual[mask] - validation_predicted[mask])
        if residuals.size < 3 and fitted is not None:
            fitted = np.asarray(fitted, dtype=float)
            y = np.asarray(y, dtype=float)
            mask = np.isfinite(y) & np.isfinite(fitted)
            residuals = np.abs(y[mask] - fitted[mask])
        if residuals.size < 3:
            residuals = np.abs(np.diff(np.asarray(y, dtype=float)))
            residuals = residuals[np.isfinite(residuals)]
        if residuals.size == 0:
            radius = np.full(len(future), 1.0)
        else:
            q = float(np.nanquantile(residuals, 0.95))
            if not np.isfinite(q) or q == 0:
                q = float(np.nanstd(residuals)) or 1.0
            radius = q * np.sqrt(np.arange(1, len(future) + 1))
        return future - radius, future + radius

    def _residual_diagnostics(y, fitted):
        y = np.asarray(y, dtype=float)
        fitted = np.asarray(fitted, dtype=float)
        mask = np.isfinite(y) & np.isfinite(fitted)
        residuals = y[mask] - fitted[mask]
        diagnostics = {
            "n": int(len(residuals)),
            "mean_residual": np.nan,
            "residual_std": np.nan,
            "bias": "N/A",
            "ljung_box_p": np.nan,
            "normality_p": np.nan,
            "residuals": residuals,
        }
        if len(residuals) == 0:
            return diagnostics

        diagnostics["mean_residual"] = float(np.mean(residuals))
        diagnostics["residual_std"] = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
        tolerance = diagnostics["residual_std"] / max(np.sqrt(len(residuals)), 1)
        diagnostics["bias"] = "over-forecasting" if diagnostics["mean_residual"] < -tolerance else "under-forecasting" if diagnostics["mean_residual"] > tolerance else "low bias"

        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox

            lag = max(1, min(10, len(residuals) // 3))
            lb = acorr_ljungbox(residuals, lags=[lag], return_df=True)
            diagnostics["ljung_box_p"] = float(lb["lb_pvalue"].iloc[-1])
        except Exception:
            pass

        try:
            from scipy import stats as scipy_stats

            if len(residuals) >= 8:
                diagnostics["normality_p"] = float(scipy_stats.normaltest(residuals).pvalue)
        except Exception:
            pass
        return diagnostics

    def _detect_anomalies(y, axis=None):
        values = pd.Series(pd.to_numeric(y, errors="coerce")).astype(float)
        if values.dropna().empty:
            return pd.DataFrame(columns=["Index", "Axis", "Actual", "Score", "Reason"])

        rolling_window = max(5, min(21, len(values) // 5 or 5))
        rolling_median = values.rolling(rolling_window, center=True, min_periods=3).median().fillna(values.median())
        residual = values - rolling_median
        mad = float(np.nanmedian(np.abs(residual - np.nanmedian(residual))))
        robust_sigma = 1.4826 * mad if mad else float(np.nanstd(residual)) or 1.0
        robust_z = residual / robust_sigma

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flags = (robust_z.abs() >= 3.5) | (values < lower) | (values > upper)

        axis_values = list(axis) if axis is not None else list(range(1, len(values) + 1))
        rows = []
        for idx in np.where(flags.fillna(False))[0]:
            reasons = []
            if abs(robust_z.iloc[idx]) >= 3.5:
                reasons.append("rolling residual")
            if values.iloc[idx] < lower or values.iloc[idx] > upper:
                reasons.append("IQR")
            axis_value = axis_values[idx] if idx < len(axis_values) else idx + 1
            rows.append(
                {
                    "Index": int(idx + 1),
                    "Axis": axis_value.strftime("%Y-%m-%d") if hasattr(axis_value, "strftime") else axis_value,
                    "Actual": float(values.iloc[idx]),
                    "Score": float(robust_z.iloc[idx]),
                    "Reason": ", ".join(reasons),
                }
            )
        return pd.DataFrame(rows)

    def _fit_arima_family(values, horizon, model_name, seasonal_period=1):
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y = np.asarray(values, dtype=float)
        notes = []

        if model_name == "SARIMA":
            period = max(2, int(seasonal_period or 1))
            if len(y) <= period * 2:
                period = max(2, min(4, len(y) // 2))
                notes.append(f"Seasonal period adjusted to {period} for the available history.")
            fit = SARIMAX(
                y,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, period),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        else:
            candidate_orders = [
                (1, 1, 1),
                (1, 0, 1),
                (2, 1, 1),
                (1, 1, 2),
                (0, 1, 1),
                (1, 1, 0),
                (0, 0, 0),
            ]
            best_fit = None
            best_aic = np.inf
            for order in candidate_orders:
                try:
                    candidate = ARIMA(y, order=order).fit()
                    if candidate.aic < best_aic:
                        best_fit = candidate
                        best_aic = candidate.aic
                except Exception:
                    continue
            if best_fit is None:
                raise RuntimeError("Unable to fit an ARIMA model to the selected response variable.")
            fit = best_fit
            if model_name in {"ARFIMA", "State Space ARIMA", "ARCH", "GARCH"}:
                notes.append(f"{model_name} is approximated with an ARIMA mean model in this PyShiny build.")

        fitted = np.asarray(fit.fittedvalues, dtype=float)
        forecast_obj = fit.get_forecast(steps=horizon)
        future = np.asarray(forecast_obj.predicted_mean, dtype=float)
        try:
            conf = forecast_obj.conf_int(alpha=0.05)
            lower = np.asarray(conf.iloc[:, 0] if hasattr(conf, 'iloc') else conf[:, 0], dtype=float)
            upper = np.asarray(conf.iloc[:, 1] if hasattr(conf, 'iloc') else conf[:, 1], dtype=float)
        except Exception:
            lower = upper = None
        summary = str(fit.summary())
        if notes:
            summary = "\n".join(notes) + "\n\n" + summary
        return fitted, future, summary, lower, upper

    def _fit_ets(values, horizon, seasonal_period=1):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        y = np.asarray(values, dtype=float)
        seasonal = "add" if seasonal_period > 1 and len(y) >= seasonal_period * 2 else None
        fit = ExponentialSmoothing(
            y,
            trend="add" if len(y) >= 4 else None,
            seasonal=seasonal,
            seasonal_periods=seasonal_period if seasonal else None,
            initialization_method="estimated",
        ).fit(optimized=True)
        return np.asarray(fit.fittedvalues, dtype=float), np.asarray(fit.forecast(horizon), dtype=float), str(fit.summary()), None, None

    def _fit_baseline_model(values, horizon, model_name, seasonal_period=1):
        y = np.asarray(values, dtype=float)
        n = len(y)
        if model_name == "Seasonal Naive":
            period = max(1, int(seasonal_period or 1))
            if period <= 1 or n <= period:
                fitted, future, summary, _, _ = _fit_baseline_model(y, horizon, "Naive", seasonal_period)
                return fitted, future, "Seasonal naive fell back to naive because there is not enough seasonal history.", None, None
            fitted = np.full(n, np.nan)
            fitted[period:] = y[:-period]
            repeats = np.resize(y[-period:], horizon)
            return fitted, repeats.astype(float), f"Model: Seasonal Naive\nSeasonal period: {period}", None, None

        if model_name == "Moving Average":
            window = max(2, min(n, int(seasonal_period or 3) if seasonal_period and seasonal_period > 1 else 3))
            fitted = np.full(n, np.nan)
            for idx in range(1, n):
                start = max(0, idx - window)
                fitted[idx] = np.mean(y[start:idx])
            history = list(y)
            future = []
            for _ in range(horizon):
                next_value = float(np.mean(history[-window:]))
                future.append(next_value)
                history.append(next_value)
            return fitted, np.asarray(future, dtype=float), f"Model: Moving Average\nWindow: {window}", None, None

        if model_name == "Drift":
            slope = (y[-1] - y[0]) / max(n - 1, 1)
            fitted = y[0] + slope * np.arange(n)
            future = np.asarray([y[-1] + slope * (step + 1) for step in range(horizon)], dtype=float)
            return fitted, future, f"Model: Drift\nSlope per period: {slope:.6g}", None, None

        fitted = np.full(n, np.nan)
        fitted[1:] = y[:-1]
        future = np.repeat(y[-1], horizon).astype(float)
        return fitted, future, "Model: Naive\nForecast repeats the last observed value.", None, None

    def _lagged_matrix(values, lags, seasonal_period=1):
        y = np.asarray(values, dtype=float)
        x_rows = []
        y_rows = []
        feature_names = [f"lag_{lag}" for lag in range(1, lags + 1)]
        feature_names += ["rolling_mean_3", "rolling_mean_6", "rolling_mean_12", "rolling_std_6", "trend_index"]
        if seasonal_period and seasonal_period > 1:
            feature_names += ["season_sin", "season_cos"]

        def features_at(history, index):
            row = [history[-lag] if len(history) >= lag else history[0] for lag in range(1, lags + 1)]
            for window in (3, 6, 12):
                window_values = history[-min(window, len(history)):]
                row.append(float(np.mean(window_values)))
            std_values = history[-min(6, len(history)):]
            row.append(float(np.std(std_values, ddof=1)) if len(std_values) > 1 else 0.0)
            row.append(float(index))
            if seasonal_period and seasonal_period > 1:
                angle = 2 * np.pi * (index % seasonal_period) / seasonal_period
                row.extend([float(np.sin(angle)), float(np.cos(angle))])
            return row

        for index in range(lags, len(y)):
            x_rows.append(features_at(y[:index], index))
            y_rows.append(y[index])
        return np.asarray(x_rows), np.asarray(y_rows), feature_names, features_at

    def _fit_lagged_regressor(values, horizon, model_name, seasonal_period=1):
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        y = np.asarray(values, dtype=float)
        lags = max(1, min(12, len(y) // 3))
        x_train, y_train, feature_names, features_at = _lagged_matrix(y, lags, seasonal_period)
        if len(x_train) < 2:
            raise RuntimeError("Not enough observations to train the selected machine-learning forecaster.")

        if model_name == "GRNN":
            estimator = make_pipeline(StandardScaler(), KernelRidge(kernel="rbf", alpha=0.2))
        elif model_name == "Neural Network":
            estimator = make_pipeline(
                StandardScaler(),
                MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42),
            )
        elif model_name == "AutoML":
            estimator = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=2)
        else:
            estimator = GradientBoostingRegressor(random_state=42)

        estimator.fit(x_train, y_train)
        fitted = np.full(len(y), np.nan)
        fitted[lags:] = estimator.predict(x_train)

        history = list(y[-lags:])
        full_history = list(y)
        future = []
        for step in range(horizon):
            row = np.asarray(features_at(np.asarray(full_history, dtype=float), len(y) + step)).reshape(1, -1)
            next_value = float(estimator.predict(row)[0])
            future.append(next_value)
            history.append(next_value)
            full_history.append(next_value)

        summary = [
            f"Model: {model_name}",
            f"Lag features: {lags}",
            "Engineered features: lags, rolling means, rolling volatility, trend, and seasonal Fourier terms when seasonality is detected.",
            f"Estimator: {estimator}",
        ]
        final_estimator = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
        importance = None
        if hasattr(final_estimator, "feature_importances_"):
            importance = {"features": feature_names, "importance": final_estimator.feature_importances_.tolist()}
        return fitted, np.asarray(future, dtype=float), "\n".join(summary), None, None, importance

    def _fit_prophet(values, horizon, time_index=None, frequency=None):
        from prophet import Prophet

        y = np.asarray(values, dtype=float)
        ds = None
        if time_index is not None:
            raw_time = pd.Series(time_index)
            raw_numeric = pd.to_numeric(raw_time, errors="coerce")
            looks_numeric = raw_numeric.notna().sum() >= max(2, int(len(raw_time) * 0.8))
            parsed = pd.to_datetime(raw_time, errors="coerce") if not looks_numeric else pd.Series([pd.NaT] * len(raw_time))
            if parsed.notna().sum() == len(y):
                ds = parsed.iloc[: len(y)].reset_index(drop=True)
        if ds is None:
            ds = pd.date_range("2000-01-01", periods=len(y), freq="D")
        model_df = pd.DataFrame(
            {
                "ds": ds,
                "y": y,
            }
        )
        model = Prophet()
        model.fit(model_df)
        future_frame = model.make_future_dataframe(periods=horizon, freq=frequency or "D")
        predicted = model.predict(future_frame)
        future = predicted["yhat"].to_numpy()
        lower = predicted["yhat_lower"].to_numpy()
        upper = predicted["yhat_upper"].to_numpy()
        summary = [
            "Model: Prophet",
            f"Changepoints: {len(model.changepoints)}",
            f"Seasonality mode: {model.seasonality_mode}",
            f"Growth: {model.growth}",
        ]
        return future[: len(y)], future[len(y):], "\n".join(summary), lower[len(y):], upper[len(y):]

    def _fit_time_series_model(model_name, values, horizon, seasonal_period, time_index=None, frequency=None):
        if model_name in {"Naive", "Seasonal Naive", "Moving Average", "Drift"}:
            return (*_fit_baseline_model(values, horizon, model_name, seasonal_period), None)
        if model_name == "ETS":
            return (*_fit_ets(values, horizon, seasonal_period), None)
        if model_name == "Prophet":
            return (*_fit_prophet(values, horizon, time_index=time_index, frequency=frequency), None)
        if model_name in {"GRNN", "Neural Network", "AutoML"}:
            fitted, future, summary, lower, upper, importance = _fit_lagged_regressor(values, horizon, model_name, seasonal_period)
            return fitted, future, summary, lower, upper, importance
        return (*_fit_arima_family(values, horizon, model_name, seasonal_period), None)

    def _read_positive_int(input_value, default, minimum=1):
        try:
            return max(minimum, int(input_value or default))
        except (TypeError, ValueError):
            return default

    def _higher_is_better(metric):
        return metric in {"R2", "Accuracy", "Coverage"}

    def _metric_is_better(score, best_score, metric):
        if pd.isna(score):
            return False
        if _higher_is_better(metric):
            return score > best_score
        return score < best_score

    def _select_best_model(models_dict, metric, fallback_metric="MAPE"):
        if not models_dict:
            return None
        metric = metric if any(metric in data["metrics"] and not pd.isna(data["metrics"].get(metric)) for data in models_dict.values()) else fallback_metric
        best_model = None
        best_score = -np.inf if _higher_is_better(metric) else np.inf
        for model_name, model_data in models_dict.items():
            score = model_data["metrics"].get(metric, np.nan)
            if _metric_is_better(score, best_score, metric):
                best_score = score
                best_model = model_name
        return best_model or list(models_dict.keys())[0]

    def _rank_models(models_dict, metric, fallback_metric="MAPE"):
        metric = metric if any(metric in data["metrics"] and not pd.isna(data["metrics"].get(metric)) for data in models_dict.values()) else fallback_metric

        def sort_key(item):
            score = item[1]["metrics"].get(metric, np.nan)
            if pd.isna(score):
                return np.inf
            return -score if _higher_is_better(metric) else score

        return [name for name, _ in sorted(models_dict.items(), key=sort_key)]

    def _time_series_validation(model_name, y, seasonal_period):
        validation_method = input.ts_validation()
        test_periods = _read_positive_int(input.ts_test_periods(), _valid_horizon())
        n = len(y)
        if n < 6:
            return _regression_metrics(y, np.full(n, np.nan)), "In-sample fallback: not enough history for validation.", None, None

        test_periods = min(test_periods, max(1, n // 3))

        if validation_method == "Rolling Cross-Validation":
            folds = _read_positive_int(input.rolling_folds(), 4, minimum=2)
            initial_pct = float(input.rolling_initial_pct() or 60) / 100.0
            min_train = max(3, int(np.ceil(n * initial_pct)), seasonal_period * 2 if seasonal_period > 1 else 3)
            max_folds = max(1, (n - min_train) // test_periods)
            folds = min(folds, max_folds)
            if folds < 2:
                validation_method = "Standard (Train/Test)"
            else:
                actual_parts = []
                predicted_parts = []
                starts = range(n - folds * test_periods, n, test_periods)
                for start in starts:
                    train_y = y[:start]
                    test_y = y[start:start + test_periods]
                    if len(train_y) < 3 or len(test_y) == 0:
                        continue
                    try:
                        _, forecast, _, _, _, _ = _fit_time_series_model(model_name, train_y, len(test_y), seasonal_period)
                    except Exception:
                        continue
                    actual_parts.append(test_y)
                    predicted_parts.append(forecast[: len(test_y)])
                if actual_parts and predicted_parts:
                    validation_actual = np.concatenate(actual_parts)
                    validation_predicted = np.concatenate(predicted_parts)
                    metrics = _regression_metrics(validation_actual, validation_predicted, training_series=y[: n - len(validation_actual)])
                    return metrics, f"Rolling expanding-window validation: {len(actual_parts)} folds, {test_periods} periods per fold.", validation_actual, validation_predicted

        train_y = y[:-test_periods]
        test_y = y[-test_periods:]
        if len(train_y) < 3 or len(test_y) == 0:
            return _regression_metrics(y, np.full(n, np.nan)), "In-sample fallback: not enough history after holdout.", None, None
        try:
            _, forecast, _, _, _, _ = _fit_time_series_model(model_name, train_y, len(test_y), seasonal_period)
            metrics = _regression_metrics(test_y, forecast[: len(test_y)], training_series=train_y)
            return metrics, f"Last-{len(test_y)} holdout validation.", test_y, forecast[: len(test_y)]
        except Exception:
            return _regression_metrics(y, np.full(n, np.nan)), "Validation fallback: model could not refit on the training window.", None, None

    def _run_time_series_forecast(df, response_column):
        horizon = _valid_horizon()
        model_names = input.model()
        if not model_names:
            model_names = ["ARIMA"]
        elif isinstance(model_names, str):
            model_names = [model_names]

        series = pd.to_numeric(df[response_column], errors="coerce")
        if input.preprocess_interpolate():
            series = series.interpolate(method="linear", limit_direction="both")

        model_df = df.loc[series.notna()].copy().reset_index(drop=True)
        y = series.dropna().astype(float).to_numpy()

        if input.preprocess_outliers():
            q1 = np.percentile(y, 25)
            q3 = np.percentile(y, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            y = np.clip(y, lower_bound, upper_bound)
        if len(y) < 3:
            return _as_error("Select a numeric response variable with at least 3 valid observations.")

        profile = _time_series_profile(model_df, response_column)
        detected_period = profile.get("detected_period")
        seasonal_period = int(input.seasonal_period() or detected_period or 1) if input.seasonal() else int(detected_period or 1)

        actual_axis, future_axis, x_label = _future_axis_from_time(model_df, horizon)
        
        selected_metric = input.best_model_metric()
        ensemble_requested = "Ensemble" in model_names
        base_model_names = [m for m in model_names if m != "Ensemble"]
        base_model_names = list(dict.fromkeys(base_model_names))
        if ensemble_requested and not base_model_names:
            return _as_error("Select at least one base model along with Ensemble. The app only runs models selected in the settings bar.")

        models_dict = {}
        failed_models = {}
        for model_name in base_model_names:
            try:
                fitted, future, summary, lower, upper, importance = _fit_time_series_model(
                    model_name,
                    y,
                    horizon,
                    seasonal_period,
                    time_index=actual_axis,
                    frequency=profile.get("frequency"),
                )
            except Exception as exc:
                failed_models[model_name] = str(exc)
                continue

            metrics, validation_note, validation_actual, validation_predicted = _time_series_validation(model_name, y, seasonal_period)

            if lower is None or upper is None:
                lower, upper = _conformal_interval(
                    future,
                    y,
                    fitted=fitted,
                    validation_actual=validation_actual,
                    validation_predicted=validation_predicted,
                )

            if validation_actual is not None and validation_predicted is not None and len(validation_actual) == len(validation_predicted):
                val_lower, val_upper = _conformal_interval(validation_predicted, y, fitted=fitted)
                metrics = _regression_metrics(
                    validation_actual,
                    validation_predicted,
                    training_series=y,
                    lower=val_lower,
                    upper=val_upper,
                )

            diagnostics = _residual_diagnostics(y, fitted)

            models_dict[model_name] = {
                "fitted": fitted,
                "future": future,
                "summary": f"{validation_note}\n\nPrediction intervals use native model intervals when available; otherwise they use conformal residual quantiles.\n\n{summary}",
                "metrics": metrics,
                "lower": lower,
                "upper": upper,
                "validation_actual": validation_actual,
                "validation_predicted": validation_predicted,
                "diagnostics": diagnostics,
                "importance": importance,
            }

        if ensemble_requested and models_dict:
            ranked_names = _rank_models(models_dict, selected_metric)
            benchmark_score = models_dict.get("Naive", {}).get("metrics", {}).get(selected_metric, np.nan)
            if not pd.isna(benchmark_score):
                ranked_names = [
                    name
                    for name in ranked_names
                    if name == "Naive" or _metric_is_better(models_dict[name]["metrics"].get(selected_metric, np.nan), benchmark_score, selected_metric)
                ] or _rank_models(models_dict, selected_metric)
            top_names = [name for name in ranked_names if name != "Ensemble"][:3]
            if not top_names:
                return _as_error("Select at least one base model along with Ensemble.")
            top_models = [models_dict[name] for name in top_names]
            all_fitted = np.vstack([np.asarray(m["fitted"], dtype=float) for m in top_models])
            all_future = np.vstack([np.asarray(m["future"], dtype=float) for m in top_models])

            scores = np.asarray([models_dict[name]["metrics"].get(selected_metric, np.nan) for name in top_names], dtype=float)
            if _higher_is_better(selected_metric):
                clean = np.where(np.isfinite(scores), np.maximum(scores, 0), 0)
                weights = clean / clean.sum() if clean.sum() else np.repeat(1 / len(top_names), len(top_names))
            else:
                clean = np.where(np.isfinite(scores) & (scores > 0), scores, np.nan)
                inv = 1 / clean
                weights = inv / np.nansum(inv) if np.isfinite(inv).any() and np.nansum(inv) else np.repeat(1 / len(top_names), len(top_names))
            weights = np.asarray(weights, dtype=float)

            ens_fitted = np.nansum(all_fitted * weights[:, None], axis=0)
            ens_future = np.nansum(all_future * weights[:, None], axis=0)

            lower_arrays = [np.asarray(m["lower"], dtype=float) for m in top_models if m["lower"] is not None]
            upper_arrays = [np.asarray(m["upper"], dtype=float) for m in top_models if m["upper"] is not None]
            ens_lower = np.nansum(np.vstack(lower_arrays) * weights[: len(lower_arrays), None], axis=0) if lower_arrays else None
            ens_upper = np.nansum(np.vstack(upper_arrays) * weights[: len(upper_arrays), None], axis=0) if upper_arrays else None

            validation_pairs = [
                (m.get("validation_actual"), m.get("validation_predicted"))
                for m in top_models
                if m.get("validation_actual") is not None and m.get("validation_predicted") is not None
            ]
            if validation_pairs and all(len(pair[1]) == len(validation_pairs[0][1]) for pair in validation_pairs):
                ens_validation_actual = validation_pairs[0][0]
                ens_validation_predicted = np.nansum(np.vstack([pair[1] for pair in validation_pairs]) * weights[: len(validation_pairs), None], axis=0)
                val_lower, val_upper = _conformal_interval(ens_validation_predicted, y, fitted=ens_fitted)
                ens_metrics = _regression_metrics(ens_validation_actual, ens_validation_predicted, training_series=y, lower=val_lower, upper=val_upper)
                validation_note = f"Weighted ensemble validation from top {len(top_names)} models: {', '.join(top_names)}."
            else:
                ens_validation_actual = None
                ens_validation_predicted = None
                ens_metrics = _regression_metrics(y, ens_fitted, training_series=y)
                validation_note = "Ensemble validation fallback used fitted values because component backtests did not align."

            models_dict["Ensemble"] = {
                "fitted": ens_fitted,
                "future": ens_future,
                "summary": validation_note + "\n\nWeights: " + ", ".join(f"{name}={weight:.2f}" for name, weight in zip(top_names, weights)),
                "metrics": ens_metrics,
                "lower": ens_lower,
                "upper": ens_upper,
                "validation_actual": ens_validation_actual,
                "validation_predicted": ens_validation_predicted,
                "diagnostics": _residual_diagnostics(y, ens_fitted),
                "importance": None,
            }

        # Scenario Adjustment Logic
        scenario_adj = float(input.scenario_adj() or 0.0)
        if scenario_adj != 0.0:
            adj_factor = 1.0 + (scenario_adj / 100.0)
            for m_data in models_dict.values():
                m_data["future"] = m_data["future"] * adj_factor
                if m_data["lower"] is not None:
                    m_data["lower"] = m_data["lower"] * adj_factor
                if m_data["upper"] is not None:
                    m_data["upper"] = m_data["upper"] * adj_factor

        if not models_dict:
            detail = "; ".join(f"{name}: {message}" for name, message in failed_models.items())
            return _as_error("Could not fit any of the selected models." + (f" Details: {detail}" if detail else ""))

        best_model = _select_best_model(models_dict, selected_metric)

        result_table = pd.DataFrame(
            {
                "Axis": list(actual_axis) + list(future_axis),
                "Actual": list(y) + [np.nan] * horizon,
            }
        )
        for m_name, m_data in models_dict.items():
            result_table[f"{m_name}_Fitted"] = list(m_data["fitted"]) + [np.nan] * horizon
            result_table[f"{m_name}_Forecast"] = [np.nan] * len(y) + list(m_data["future"])
            if m_data.get("lower") is not None and m_data.get("upper") is not None:
                result_table[f"{m_name}_Lower_95"] = [np.nan] * len(y) + list(m_data["lower"])
                result_table[f"{m_name}_Upper_95"] = [np.nan] * len(y) + list(m_data["upper"])

        return {
            "ok": True,
            "kind": "Time Series",
            "model_name": ", ".join(models_dict.keys()),
            "best_model": best_model,
            "models": models_dict,
            "response_column": response_column,
            "x_label": x_label,
            "actual_axis": actual_axis,
            "future_axis": future_axis,
            "actual": y,
            "table": result_table,
            "validation_method": input.ts_validation(),
            "test_periods": _read_positive_int(input.ts_test_periods(), _valid_horizon()),
            "scenario_adjustment": scenario_adj,
            "profile": profile,
            "anomalies": _detect_anomalies(y, actual_axis),
            "failed_models": failed_models,
        }

    def _prepared_feature_frame(df, feature_columns):
        feature_frame = df[feature_columns].copy()
        for column in feature_frame.columns:
            if pd.api.types.is_numeric_dtype(feature_frame[column]):
                feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")
                feature_frame[column] = feature_frame[column].fillna(feature_frame[column].median())
            else:
                feature_frame[column] = feature_frame[column].astype("object").fillna("Missing")
        return pd.get_dummies(feature_frame, drop_first=False)

    def _sklearn_regression_summary(model_name, model, feature_names):
        rows = [f"Model: {model_name}", f"Features: {len(feature_names)}"]
        estimator = model.steps[-1][1] if hasattr(model, "steps") else model
        if hasattr(estimator, "intercept_"):
            rows.append(f"Intercept: {np.ravel(estimator.intercept_)[0]:.6g}")
        if hasattr(estimator, "coef_"):
            coefficients = np.ravel(estimator.coef_)
            rows.append("Coefficients:")
            for name, coefficient in zip(feature_names, coefficients):
                rows.append(f"  {name}: {coefficient:.6g}")
        rows.append(f"Estimator: {model}")
        return "\n".join(rows)

    def _run_tabular_forecast(df, response_column):
        from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        import statsmodels.api as sm

        model_names = input.model1()
        if not model_names:
            model_names = ["Linear Regression"]
        elif isinstance(model_names, str):
            model_names = [model_names]
            
        feature_columns = _feature_columns(df, response_column)
        if not feature_columns:
            return _as_error("Select at least one predictor variable for non-time-series forecasting.")

        y_raw = df[response_column]
        y_numeric = pd.to_numeric(y_raw, errors="coerce")
        keep_rows = y_raw.notna() & y_numeric.notna() if "Logistic Regression" not in model_names else y_raw.notna()

        model_df = df.loc[keep_rows].reset_index(drop=True)
        if model_df.empty:
            return _as_error("No complete rows are available for the selected response variable.")

        x_frame = _prepared_feature_frame(model_df, feature_columns)
        if x_frame.empty:
            return _as_error("The selected predictor variables could not be prepared for modeling.")
        x_frame = x_frame.astype(float)

        use_backtesting = input.use_backtesting()
        train_pct = input.train_split() / 100.0 if use_backtesting else 1.0

        if use_backtesting and len(model_df) > 5:
            indices = np.arange(len(model_df))
            if input.tabular_split_mode() == "Last N Rows":
                test_rows = _read_positive_int(input.tabular_test_rows(), max(1, int(len(model_df) * (1 - train_pct))))
                test_rows = min(test_rows, len(model_df) - 2)
                idx_train = indices[:-test_rows]
                idx_test = indices[-test_rows:]
            else:
                idx_train, idx_test = train_test_split(indices, train_size=train_pct, random_state=42)
                idx_train = np.sort(idx_train)
                idx_test = np.sort(idx_test)
        else:
            idx_train = np.arange(len(model_df))
            idx_test = idx_train

        x_train = x_frame.iloc[idx_train]
        x_test = x_frame.iloc[idx_test]

        models_dict = {}
        failed_models = {}
        metric_kind = "regression"
        for model_name in model_names:
            try:
                if model_name == "Logistic Regression" and y_raw.loc[keep_rows].nunique(dropna=True) <= 20:
                    y = y_raw.loc[keep_rows].reset_index(drop=True)
                    y_tr, y_te = y.iloc[idx_train], y.iloc[idx_test]
                    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
                    model.fit(x_train, y_tr)
                    fitted_full = model.predict(x_frame)
                    metrics = _classification_metrics(y_te, model.predict(x_test))
                    summary = _sklearn_regression_summary(model_name, model, x_frame.columns.tolist())
                    metric_kind = "classification"
                    
                    estimator = model.steps[-1][1]
                    coef = estimator.coef_[0] if len(estimator.coef_) == 1 else estimator.coef_[0]
                    importance = {"features": x_frame.columns.tolist(), "importance": coef.tolist()}
                else:
                    if model_name == "Logistic Regression":
                        failed_models[model_name] = "Logistic Regression requires a categorical target with 20 or fewer classes."
                        continue
                    y = y_numeric.loc[keep_rows].astype(float).reset_index(drop=True)
                    y_tr, y_te = y.iloc[idx_train], y.iloc[idx_test]
                    metric_kind = "regression"
                    importance = None
                    if model_name == "GLM":
                        x_tr_const = sm.add_constant(x_train, has_constant="add")
                        x_full_const = sm.add_constant(x_frame, has_constant="add")
                        model = sm.GLM(y_tr, x_tr_const, family=sm.families.Gaussian()).fit()
                        fitted_full = model.predict(x_full_const)
                        metrics = _regression_metrics(y_te, model.predict(sm.add_constant(x_test, has_constant="add")))
                        summary = str(model.summary())
                        if 'const' in model.params:
                            importance = {"features": list(x_frame.columns), "importance": model.params.drop('const').tolist()}
                        else:
                            importance = {"features": list(x_frame.columns), "importance": model.params.tolist()}
                    elif model_name == "LASSO":
                        model = make_pipeline(StandardScaler(), Lasso(alpha=0.01, max_iter=10000))
                        model.fit(x_train, y_tr)
                        fitted_full = model.predict(x_frame)
                        metrics = _regression_metrics(y_te, model.predict(x_test))
                        summary = _sklearn_regression_summary(model_name, model, x_frame.columns.tolist())
                        importance = {"features": x_frame.columns.tolist(), "importance": model.steps[-1][1].coef_.tolist()}
                    elif model_name == "Ridge Regression":
                        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                        model.fit(x_train, y_tr)
                        fitted_full = model.predict(x_frame)
                        metrics = _regression_metrics(y_te, model.predict(x_test))
                        summary = _sklearn_regression_summary(model_name, model, x_frame.columns.tolist())
                        importance = {"features": x_frame.columns.tolist(), "importance": model.steps[-1][1].coef_.tolist()}
                    else:
                        model = LinearRegression()
                        model.fit(x_train, y_tr)
                        fitted_full = model.predict(x_frame)
                        metrics = _regression_metrics(y_te, model.predict(x_test))
                        summary = _sklearn_regression_summary("Linear Regression", model, x_frame.columns.tolist())
                        importance = {"features": x_frame.columns.tolist(), "importance": model.coef_.tolist()}
            except Exception:
                failed_models[model_name] = "Model fitting failed. Check variable types, missing values, or target suitability."
                continue

            models_dict[model_name] = {
                "fitted": fitted_full,
                "summary": summary,
                "metrics": metrics,
                "importance": importance,
                "estimator": model,
            }

        if not models_dict:
            detail = "; ".join(f"{name}: {message}" for name, message in failed_models.items())
            return _as_error("Could not fit any of the selected models." + (f" Details: {detail}" if detail else ""))

        best_metric = input.best_model_metric()
        if metric_kind == "classification":
            best_metric = "Accuracy"
        
        best_model = _select_best_model(models_dict, best_metric, fallback_metric="Accuracy" if metric_kind == "classification" else "MAPE")

        scenario_predictions = {}
        scenario_feature = input.scenario_feature()
        scenario_value = input.scenario_value()
        if input.use_scenario() and scenario_feature in feature_columns and scenario_value not in (None, ""):
            scenario_row = model_df[feature_columns].tail(1).copy()
            if pd.api.types.is_numeric_dtype(model_df[scenario_feature]):
                parsed_value = pd.to_numeric(pd.Series([scenario_value]), errors="coerce").iloc[0]
                if not pd.isna(parsed_value):
                    scenario_row.loc[:, scenario_feature] = parsed_value
            else:
                scenario_row.loc[:, scenario_feature] = str(scenario_value)
            scenario_x = _prepared_feature_frame(scenario_row, feature_columns).reindex(columns=x_frame.columns, fill_value=0).astype(float)
            for m_name, m_data in models_dict.items():
                estimator = m_data.get("estimator")
                if estimator is None:
                    continue
                try:
                    if m_name == "GLM":
                        prediction = estimator.predict(sm.add_constant(scenario_x, has_constant="add"))
                    else:
                        prediction = estimator.predict(scenario_x)
                    scenario_predictions[m_name] = np.asarray(prediction).ravel()[0]
                except Exception:
                    scenario_predictions[m_name] = np.nan

        axis = np.arange(1, len(model_df) + 1)
        result_table = pd.DataFrame({
            "Sequence": axis,
            "Actual": list(y_raw.loc[keep_rows]) if metric_kind == "classification" else list(y_numeric.loc[keep_rows]),
        })
        for m_name, m_data in models_dict.items():
            result_table[f"{m_name}_Forecast"] = list(m_data["fitted"])
        if scenario_predictions:
            scenario_row = {"Sequence": "Scenario", "Actual": np.nan}
            for m_name in models_dict:
                scenario_row[f"{m_name}_Forecast"] = scenario_predictions.get(m_name, np.nan)
            result_table = pd.concat([result_table, pd.DataFrame([scenario_row])], ignore_index=True)

        return {
            "ok": True,
            "kind": "Non-Time Series",
            "model_name": ", ".join(models_dict.keys()),
            "best_model": best_model,
            "models": models_dict,
            "response_column": response_column,
            "x_label": "Sequence",
            "actual_axis": axis,
            "future_axis": [],
            "actual": np.asarray(y_raw.loc[keep_rows] if metric_kind == "classification" else y_numeric.loc[keep_rows]),
            "metric_kind": metric_kind,
            "table": result_table,
            "validation_method": input.tabular_split_mode() if use_backtesting else "In-sample fit",
            "train_split": input.train_split() if use_backtesting else 100,
            "scenario_feature": scenario_feature if scenario_predictions else None,
            "scenario_value": scenario_value if scenario_predictions else None,
            "scenario_predictions": scenario_predictions,
            "failed_models": failed_models,
        }

    def _build_forecast_result():
        df = loaded_data()
        if df is None:
            return _as_error("Load data before running a forecast.")
        if df.empty:
            return _as_error("The loaded data is empty.")

        selected_response = selected_response_column()
        if selected_response is None or selected_response not in df.columns:
            return _as_error("Select a numeric response variable before running a forecast.")

        if input.data_type() == "Non-Time Series":
            return _run_tabular_forecast(df, selected_response)
        return _run_time_series_forecast(df, selected_response)

    def _detect_seasonal_period(y):
        values = pd.Series(pd.to_numeric(y, errors="coerce")).dropna().astype(float)
        if len(values) < 12 or values.std() == 0:
            return None
        candidates = [4, 5, 7, 12, 24, 30, 52]
        best_period = None
        best_corr = 0
        for period in candidates:
            if len(values) <= period * 2:
                continue
            corr = values.autocorr(lag=period)
            if pd.notna(corr) and corr > best_corr:
                best_corr = corr
                best_period = period
        try:
            from scipy.signal import periodogram

            centered = values.to_numpy() - values.mean()
            freqs, power = periodogram(centered)
            valid = freqs > 0
            if valid.any():
                periods = 1 / freqs[valid]
                power = power[valid]
                strongest = float(periods[np.argmax(power)])
                rounded = int(round(strongest))
                if 2 <= rounded <= len(values) // 2:
                    corr = values.autocorr(lag=rounded)
                    if pd.notna(corr) and corr > best_corr:
                        best_corr = corr
                        best_period = rounded
        except Exception:
            pass
        return best_period if best_corr >= 0.3 else None

    @reactive.effect
    @reactive.event(input.implement_forecasting, ignore_init=True)
    def _implement_forecasting_from_summary():
        df = loaded_data()
        if df is None or df.empty or not _numeric_columns(df):
            ui.notification_show("Load data with at least one numeric column before forecasting.", type="warning", duration=4)
            return
        ui.update_navset("main_nav", selected="forecasting", session=session)

    def _recommended_settings(df, data_type, response_column):
        if df is None or df.empty:
            return None

        quality = data_quality_summary()
        if quality is None:
            return None
        if response_column not in _numeric_columns(df):
            return None

        if data_type == "Time Series":
            y = pd.to_numeric(df[response_column], errors="coerce").dropna()
            profile = _time_series_profile(df, response_column)
            detected_period = profile.get("detected_period")
            frequency_label = profile.get("frequency_label", "series")
            if str(frequency_label).lower() in {"n", "ns", "us", "ms"}:
                frequency_label = "series"
            if len(y) < 12:
                models = ["Naive", "Moving Average", "ETS"]
                reason = "short history"
            elif detected_period:
                models = ["Seasonal Naive", "SARIMA", "Prophet", "Ensemble"]
                reason = f"{frequency_label} data with seasonal signal"
            elif len(y) > 80:
                models = ["ARIMA", "ETS", "AutoML", "Ensemble"]
                reason = "enough history to compare multiple models"
            else:
                models = ["Naive", "Drift", "ARIMA", "ETS"]
                reason = "moderate history without strong detected seasonality"

            profile_note = f"Trend looks {profile.get('trend', 'flat')}."
            if profile.get("notes"):
                profile_note += " " + "; ".join(profile["notes"]) + "."
            caution = " Clean data quality issues first." if any(count for _, count, _ in quality["rows"][:4]) else ""
            return {
                "response": response_column,
                "models": models,
                "reason": reason,
                "note": f"{profile_note}{caution}",
                "seasonal": bool(detected_period),
                "seasonal_period": detected_period or 12,
                "metric": "MASE",
                "details": [
                    ("History", f"{len(y):,} valid observations"),
                    ("Frequency", frequency_label),
                    ("Seasonality", detected_period or "Not detected"),
                    ("Trend", profile.get("trend", "flat")),
                    ("Validation", input.ts_validation()),
                ],
            }

        feature_cols = _feature_columns(df, response_column)
        if len(feature_cols) > len(df) / 2:
            models = ["LASSO", "Ridge Regression"]
            reason = "many predictors relative to rows"
        elif len(feature_cols) <= 2:
            models = ["Linear Regression", "GLM"]
            reason = "small feature set"
        else:
            models = ["Linear Regression", "Ridge Regression", "LASSO"]
            reason = "balanced feature and row count"
        caution = " Address missing or mixed-type columns first." if quality["non_numeric"] or quality["rows"][0][1] else ""
        return {
            "response": response_column,
            "models": models,
            "reason": reason,
            "note": caution.strip(),
            "metric": "RMSE",
            "details": [
                ("Rows", f"{len(df):,}"),
                ("Predictors", f"{len(feature_cols):,}"),
                ("Validation", input.tabular_split_mode() if input.use_backtesting() else "In-sample fit"),
                ("Recommended metric", "RMSE"),
            ],
        }

    @reactive.calc
    @reactive.event(input.response_variable, input.data_type, ignore_none=False)
    def recommended_settings():
        df = loaded_data()
        return _recommended_settings(df, input.data_type(), input.response_variable())

    @output
    @render.ui
    def model_recommendation():
        settings = recommended_settings()
        if not settings:
            return ui.div()
        suggested = ", ".join(settings["models"])
        note = f" {settings['note']}" if settings.get("note") else ""
        return ui.div(
            ui.card(
                ui.div(
                    ui.div(
                        ui.tags.i(class_="fa-solid fa-lightbulb"),
                        ui.span(" Recommended settings"),
                        class_="recommendation-heading",
                    ),
                    ui.p(f"For {settings['response']}: {suggested}", class_="recommendation-copy"),
                    ui.p(f"{settings['reason'].capitalize()}.{note}", class_="recommendation-note"),
                    class_="recommendation-content",
                ),
                class_="recommendation-card recommendation-card-main",
            ),
            ui.card(
                ui.tags.details(
                    ui.tags.summary("Why this recommendation"),
                    ui.div(
                        *(
                            ui.div(
                                ui.span(f"{label}: ", class_="recommendation-detail-label"),
                                ui.span(str(value), class_="recommendation-detail-value"),
                                class_="recommendation-detail-row",
                            )
                            for label, value in settings.get("details", [])
                        ),
                        class_="recommendation-detail-grid",
                    ),
                    class_="recommendation-details",
                ),
                class_="recommendation-card recommendation-card-details",
            ),
            ui.div(
                ui.input_action_button(
                    "apply_recommendation",
                    "Use recommended settings",
                    icon=ui.tags.i(class_="fa-solid fa-wand-magic-sparkles"),
                    class_="btn-info btn-sm w-100",
                ),
                class_="recommendation-action",
            ),
            class_="recommendation-panel",
        )

    @reactive.effect
    @reactive.event(input.apply_recommendation, ignore_init=True)
    def _apply_recommended_settings():
        settings = recommended_settings()
        if not settings:
            ui.notification_show("Load data before applying recommended settings.", type="warning", duration=4)
            return

        if input.data_type() == "Time Series":
            selected_models = [model for model in settings["models"] if model in TIME_SERIES_MODEL_CHOICES]
            ui.update_selectize("model", choices=TIME_SERIES_MODEL_CHOICES, selected=selected_models, session=session)
            ui.update_checkbox("seasonal", value=settings.get("seasonal", False), session=session)
            ui.update_numeric("seasonal_period", value=settings.get("seasonal_period", 12), session=session)
        else:
            selected_models = [model for model in settings["models"] if model in TABULAR_MODEL_CHOICES]
            ui.update_selectize("model1", choices=TABULAR_MODEL_CHOICES, selected=selected_models, session=session)
        ui.update_select("best_model_metric", selected=settings.get("metric", "MASE"), session=session)
        ui.notification_show("Recommended settings applied.", type="message", duration=3)

    @reactive.calc
    def _forecast_is_ready():
        df = loaded_data()
        if df is None or df.empty:
            return False
        response = input.response_variable()
        if response not in _numeric_columns(df):
            return False
        if input.data_type() == "Time Series":
            return bool(_normalize_selection(input.model()))
        return bool(_normalize_selection(input.model1())) and bool(selected_feature_columns())

    @output
    @render.ui
    def forecast_action_controls():
        ready = _forecast_is_ready()
        result = forecast_result.get()
        report_ready = bool(result and result.get("ok"))
        is_stale = _forecast_is_stale(result)
        return ui.div(
            ui.input_action_button(
                "forecast",
                "Regenerate Forecast" if is_stale else "Generate Forecast",
                icon=ui.tags.i(class_="fa-solid fa-arrow-right"),
                class_="btn-primary w-100",
                disabled=not ready,
            ),
            ui.download_button(
                "download",
                "Download Data",
                icon=ui.tags.i(class_="fa-solid fa-download"),
                class_="btn-info w-100 mt-2" + ("" if report_ready else " disabled"),
                disabled=not report_ready,
            ),
            ui.download_button(
                "download_report",
                "Download Full Report",
                icon=ui.tags.i(class_="fa-solid fa-file-lines"),
                class_="btn-secondary w-100 mt-2" + ("" if report_ready else " disabled"),
                disabled=not report_ready,
            ),
            class_="button-column",
        )

    @output
    @render.ui
    def report_preview():
        result = forecast_result.get()
        if result is None or not result.get("ok"):
            return ui.div()
        sections = ["Model settings", "Forecast chart", "Model comparison", "Forecast data"]
        if loaded_data() is not None:
            sections.extend(["Data quality", "Summary statistics"])
        if result.get("kind") == "Time Series":
            sections.extend(["Residual diagnostics", "Anomalies", "Forecast explainability"])
        else:
            sections.extend(["Validation summary", "Feature importance"])
        if result.get("scenario_predictions"):
            sections.append("Scenario forecast")
        return ui.div(
            ui.div("Report preview", class_="report-preview-title"),
            ui.tags.ul(*(ui.tags.li(section) for section in sections), class_="report-preview-list"),
            class_="report-preview",
        )

    @reactive.calc
    @reactive.event(input.forecast, ignore_init=True)
    def generated_forecast_result():
        if not _forecast_is_ready():
            ui.notification_show("Choose a response variable and at least one model before generating a forecast.", type="warning", duration=4)
            return _as_error("Choose a response variable and at least one model before generating a forecast.")
        signature = _forecast_signature()
        ui.notification_show("Forecast run started. Training and comparing selected models...", type="message", duration=3)
        try:
            with ui.Progress(min=0, max=1, session=session) as progress:
                progress.set(0.15, message="Preparing data", detail="Checking selected target, features, and preprocessing.")
                result = _build_forecast_result()
                progress.set(0.85, message="Finalizing output", detail="Building plots, metrics, and report data.")
        except Exception:
            result = _build_forecast_result()
        if isinstance(result, dict):
            result["signature"] = signature
        return result

    @reactive.effect
    def _store_generated_forecast_result():
        result = generated_forecast_result()
        forecast_result.set(result)
        if result.get("ok"):
            ui.notification_show("Forecast complete.", type="message", duration=3)

    @output
    @render.ui
    def forecast_status():
        result = forecast_result.get()
        if result is None:
            return ui.div(
                "Choose a response variable and model, then generate a forecast.",
                class_="alert alert-info py-2",
            )
        if not result.get("ok"):
            return ui.div(result.get("message", "Forecasting failed."), class_="alert alert-danger py-2")
        model_count = len(result.get("models", {}))
        if result.get("kind") == "Time Series":
            forecast_scope = f"{result.get('test_periods', '')} test periods"
            horizon_label = f"{len(result.get('future_axis', []))} future periods"
        else:
            forecast_scope = f"{result.get('train_split', 100)}% training split"
            horizon_label = f"{len(result.get('table', [])):,} scored rows"
        failure_count = len(result.get("failed_models", {}))
        return ui.div(
            ui.div(
                "Settings changed after this forecast. Regenerate before downloading or sharing results.",
                class_="alert alert-warning py-2",
            ) if _forecast_is_stale(result) else ui.div(),
            ui.div(
                f"{failure_count} selected model(s) failed; successful models are still shown.",
                class_="alert alert-warning py-2",
            ) if failure_count else ui.div(),
            ui.div(
                ui.div(ui.tags.i(class_="fa-solid fa-circle-check"), class_="forecast-hero-icon"),
                ui.div(
                    ui.div("Forecast Ready", class_="forecast-kicker"),
                    ui.div(result["response_column"], class_="forecast-title"),
                    ui.div(f"{result['model_name']} generated successfully.", class_="forecast-subtitle"),
                ),
                class_="forecast-hero",
            ),
            ui.div(
                ui.div(ui.div(str(model_count), class_="forecast-stat-value"), ui.div("Models", class_="forecast-stat-label"), class_="forecast-stat-card border-primary"),
                ui.div(ui.div(result.get("best_model", "N/A"), class_="forecast-stat-value is-text"), ui.div("Best Model", class_="forecast-stat-label"), class_="forecast-stat-card border-success"),
                ui.div(ui.div(result.get("validation_method", "Validation"), class_="forecast-stat-value is-text"), ui.div(forecast_scope, class_="forecast-stat-label"), class_="forecast-stat-card border-warning"),
                ui.div(ui.div(horizon_label, class_="forecast-stat-value is-text"), ui.div("Output", class_="forecast-stat-label"), class_="forecast-stat-card border-info"),
                class_="forecast-status-grid",
            ),
            class_="forecast-status-panel",
        )

    @output
    @render.ui
    def model_plot_filter():
        result = forecast_result.get()
        if result is None or not result.get("ok"):
            return ui.div()
        model_names = list(result.get("models", {}).keys())
        if not model_names:
            return ui.div()
        return ui.div(
            ui.div(
                ui.tags.i(class_="fa-solid fa-eye"),
                ui.span("Visible Models"),
                class_="plot-filter-title",
            ),
            ui.input_checkbox_group(
                "plot_models",
                None,
                choices=model_names,
                selected=model_names,
                inline=True,
            ),
            class_="plot-filter-panel",
        )

    def _forecast_figure(result, selected_models=None):
        if result is None:
            return _empty_plotly_figure("Generate a forecast to see actual vs forecast")
        if not result.get("ok"):
            return _empty_plotly_figure(result.get("message", "Forecasting failed"))

        fig = go.Figure()
        actual_axis = result["actual_axis"]
        fig.add_trace(
            go.Scatter(
                x=actual_axis,
                y=result["actual"],
                mode="lines",
                name="Actual",
                line={"color": "#3487ff", "width": 2.3},
            )
        )

        models_dict = result.get("models", {})
        selected_set = set(selected_models) if selected_models is not None else None
        visible_model_count = len(selected_set) if selected_set is not None else len(models_dict)
        visible_model_count = max(1, visible_model_count)
        colors = ["rgb(18, 198, 163)", "rgb(245, 158, 11)", "rgb(236, 72, 153)", "rgb(139, 92, 246)", "rgb(239, 68, 68)", "rgb(59, 130, 246)"]
        color_idx = 0
        
        for m_name, m_data in models_dict.items():
            color = colors[color_idx % len(colors)]
            color_idx += 1
            if selected_set is not None and m_name not in selected_set:
                continue
            
            fitted = np.asarray(m_data.get("fitted", []))
            future = np.asarray(m_data.get("future", []))
            lower = m_data.get("lower")
            upper = m_data.get("upper")

            if len(fitted):
                fig.add_trace(
                    go.Scatter(
                        x=actual_axis,
                        y=fitted,
                        mode="lines",
                        name=f"{m_name} Fitted",
                        line={"color": color, "width": 1.8, "dash": "dot"},
                    )
                )
            if len(future):
                future_axis = result["future_axis"]
                fig.add_trace(
                    go.Scatter(
                        x=future_axis,
                        y=future,
                        mode="lines+markers",
                        name=f"{m_name} Forecast",
                        line={"color": color, "width": 2.4},
                        marker={"size": 6},
                    )
                )
                if lower is not None and upper is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=np.concatenate([future_axis, future_axis[::-1]]),
                            y=np.concatenate([upper, lower[::-1]]),
                            fill="toself",
                            fillcolor=color.replace("rgb", "rgba").replace(")", ", 0.35)"),
                            opacity=0.35,
                            line={"color": "rgba(255,255,255,0)"},
                            name=f"{m_name} 95% CI",
                            showlegend=True,
                        )
                    )

        if len(result.get("future_axis", [])) and len(actual_axis):
            split_x = actual_axis.iloc[-1] if hasattr(actual_axis, "iloc") else actual_axis[-1]
            fig.add_shape(
                type="line",
                x0=split_x,
                x1=split_x,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line={"color": "#94a3b8", "dash": "dash", "width": 1.4},
                opacity=0.65,
            )

        _plotly_dark_layout(fig, title=f"Actual vs Forecast: {result['response_column']}")
        dynamic_height = min(900, max(540, 440 + visible_model_count * 45))
        fig.update_layout(height=dynamic_height, margin={"l": 48, "r": 24, "t": 48, "b": min(220, 96 + visible_model_count * 14)})
        fig.update_xaxes(title=result.get("x_label", "Sequence"))
        fig.update_yaxes(title=result["response_column"])
        return fig

    @reactive.calc
    def selected_plot_models():
        try:
            return _normalize_selection(input.plot_models())
        except Exception:
            return None

    @output
    @render_widget
    def plot():
        return _forecast_figure(forecast_result.get(), selected_plot_models())

    @output
    @render.ui
    def forecast_plot_container():
        result = forecast_result.get()
        selected_models = selected_plot_models()

        if result is None or not result.get("ok"):
            selected_count = 1
        else:
            model_names = list(result.get("models", {}).keys())
            selected_count = len(selected_models) if selected_models is not None else len(model_names)
            selected_count = max(1, selected_count)

        height = min(900, max(540, 440 + selected_count * 45))
        return output_widget("plot", height=f"{height}px", fill=False, fillable=False)

    def _quality_verdict(result):
        if result is None or not result.get("ok"):
            return None
        models_dict = result.get("models", {})
        best_model = result.get("best_model")
        metrics = models_dict.get(best_model, {}).get("metrics", result.get("metrics", {}))
        if result.get("metric_kind") == "classification":
            accuracy = metrics.get("Accuracy", np.nan)
            if pd.notna(accuracy) and accuracy >= 85:
                return "Good fit", "Validation accuracy is strong for the selected split.", "success"
            if pd.notna(accuracy) and accuracy >= 65:
                return "Use with caution", "Validation accuracy is moderate; compare with business tolerance.", "warning"
            return "Weak validation", "Accuracy is low or unavailable. Review features, target quality, and split method.", "danger"

        mase = metrics.get("MASE", np.nan)
        coverage = metrics.get("Coverage", np.nan)
        smape = metrics.get("sMAPE", np.nan)
        if pd.notna(mase) and mase < 1 and (pd.isna(coverage) or coverage >= 70):
            return "Good fit", "Best model beats a naive benchmark and interval coverage looks usable.", "success"
        if (pd.notna(mase) and mase < 1.5) or (pd.notna(smape) and smape < 25):
            return "Use with caution", "Validation is acceptable but should be checked against business tolerance.", "warning"
        return "Weak validation", "The selected model does not clearly outperform simple baselines. Try another model or clean the data.", "danger"

    @output
    @render.ui
    def forecast_quality_verdict():
        result = forecast_result.get()
        verdict = _quality_verdict(result)
        if verdict is None:
            return ui.div()
        title, detail, color = verdict
        stale_note = " Settings changed since this run; regenerate before using the verdict." if _forecast_is_stale(result) else ""
        return ui.div(
            ui.div(ui.tags.i(class_="fa-solid fa-gauge-high"), class_="quality-verdict-icon"),
            ui.div(
                ui.div(title, class_="quality-verdict-title"),
                ui.div(detail + stale_note, class_="quality-verdict-detail"),
            ),
            class_=f"quality-verdict quality-verdict-{color}",
        )

    @output
    @render.ui
    def model_accuracy():
        result = forecast_result.get()
        if result is None or not result.get("ok"):
            return ui.div()
            
        models_dict = result.get("models", {})
        best_model = result.get("best_model", "")
        best_metric = input.best_model_metric()
        
        metric_keys = ["MASE", "WAPE", "sMAPE", "MAPE", "RMSE", "MAE", "R2", "BIC", "Coverage"] if result.get("metric_kind") != "classification" else ["Accuracy"]
        
        table_html = "<div class='table-responsive mt-2 mb-4'><table class='table table-hover table-borderless align-middle' style='border: 1px solid var(--ops-border); border-radius: 8px; overflow: hidden; background: var(--ops-panel-deep);'><thead style='background: rgba(255,255,255,0.03);'><tr><th style='padding: 12px 16px;'>Model</th>"
        for k in metric_keys:
            table_html += f"<th style='padding: 12px 16px; color: #9db2ce; font-size: 0.8rem; font-weight: 800;'>{k}</th>"
        table_html += "</tr></thead><tbody>"
        
        for m_name, m_data in models_dict.items():
            is_best = (m_name == best_model)
            row_style = "background: rgba(245, 158, 11, 0.1);" if is_best else ""
            badge = " <span class='badge' style='background: var(--ops-amber); color: #000; margin-left: 8px;'>Best</span>" if is_best else ""
            table_html += f"<tr style='{row_style}'><td style='padding: 12px 16px; font-weight: 700;'>{m_name}{badge}</td>"
            for k in metric_keys:
                val = m_data["metrics"].get(k, np.nan)
                text_color = "color: #fff;" if is_best else "color: #dbeafe;"
                suffix = "%" if k in ["MAPE", "sMAPE", "WAPE", "MdAPE", "Coverage", "Accuracy"] else ""
                table_html += f"<td style='padding: 12px 16px; {text_color}'>{_format_metric(val, suffix)}</td>"
            table_html += "</tr>"
        table_html += "</tbody></table></div>"

        if best_model and best_model in models_dict:
            metrics = models_dict[best_model]["metrics"]
        else:
            metrics = result.get("metrics", {})
            
        if result.get("metric_kind") == "classification":
            cards = [("Accuracy", _format_metric(metrics.get("Accuracy"), "%"), "bullseye", "primary")]
        else:
            cards = [
                ("MASE", _format_metric(metrics.get("MASE")), "scale-balanced", "primary"),
                ("WAPE", _format_metric(metrics.get("WAPE"), "%"), "percentage", "info"),
                ("sMAPE", _format_metric(metrics.get("sMAPE"), "%"), "chart-simple", "success"),
                ("RMSE", _format_metric(metrics.get("RMSE")), "square-root-variable", "success"),
                ("MAE", _format_metric(metrics.get("MAE")), "chart-simple", "warning"),
            ]

        return ui.div(
            ui.div("Model Comparison", style="color: #9db2ce; font-size: 0.85rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;"),
            ui.HTML(table_html),
            ui.div(
                ui.span("Best Model Metrics", style="color: #9db2ce; font-size: 0.85rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; margin-right: 12px;"),
                ui.span(best_model, class_="badge", style="background: var(--ops-accent); color: #fff; font-size: 0.85rem; padding: 0.4rem 0.6rem;"),
                class_="d-flex align-items-center mb-2 mt-4"
            ),
            ui.div(
                *(
                    ui.div(
                        ui.div(ui.tags.i(class_=f"fa-solid fa-{icon}"), class_="metric-icon"),
                        ui.div(value, class_="metric-value"),
                        ui.div(label, class_="metric-label"),
                        class_=f"metric-card border-{color}",
                    )
                    for label, value, icon, color in cards
                ),
                class_="metric-grid mt-2",
            )
        )

    @output
    @render.ui
    def validation_summary():
        result = forecast_result.get()
        if result is None:
            return ui.div("Generate a forecast to see validation details.", class_="alert alert-info py-2")
        if not result.get("ok"):
            return ui.div(result.get("message", "Forecasting failed."), class_="alert alert-danger py-2")
        if result.get("kind") == "Time Series":
            return ui.div("Backtesting is shown for time-series forecasts.", class_="alert alert-info py-2")
        split = result.get("train_split", 100)
        method = result.get("validation_method", "Validation")
        return ui.div(
            ui.div("Validation Summary", style="color: #9db2ce; font-size: 0.85rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;"),
            ui.div(
                ui.div(ui.div(str(method), class_="forecast-stat-value is-text"), ui.div("Method", class_="forecast-stat-label"), class_="forecast-stat-card border-primary"),
                ui.div(ui.div(f"{split}%", class_="forecast-stat-value is-text"), ui.div("Training Split", class_="forecast-stat-label"), class_="forecast-stat-card border-success"),
                ui.div(ui.div(result.get("best_model", "N/A"), class_="forecast-stat-value is-text"), ui.div("Best Model", class_="forecast-stat-label"), class_="forecast-stat-card border-warning"),
                class_="forecast-status-grid",
            ),
        )

    @output
    @render.text
    def fitted_model():
        result = forecast_result.get()
        if result is None:
            return "Generate a forecast to see fitted model parameters."
        if not result.get("ok"):
            return result.get("message", "Forecasting failed.")
        
        models_dict = result.get("models", {})
        if not models_dict:
            return result.get("summary", "No fitted model summary is available.")
        
        summaries = []
        for m_name, m_data in models_dict.items():
            summaries.append(f"--- {m_name} ---\n{m_data.get('summary', '')}")
        failed_models = result.get("failed_models", {})
        if failed_models:
            summaries.append(
                "--- Models that did not run ---\n"
                + "\n".join(f"{name}: {message}" for name, message in failed_models.items())
            )
        return "\n\n".join(summaries)

    def _best_time_series_model_data(result):
        if result is None or not result.get("ok") or result.get("kind") != "Time Series":
            return None, None
        models_dict = result.get("models", {})
        best_model = result.get("best_model")
        if best_model in models_dict:
            return best_model, models_dict[best_model]
        if models_dict:
            first_name = next(iter(models_dict))
            return first_name, models_dict[first_name]
        return None, None

    @output
    @render.ui
    def residual_diagnostics():
        result = forecast_result.get()
        model_name, model_data = _best_time_series_model_data(result)
        if not model_data:
            return ui.div("Generate a time-series forecast to see residual diagnostics.", class_="alert alert-info py-2")

        diagnostics = model_data.get("diagnostics", {})
        lb_p = diagnostics.get("ljung_box_p", np.nan)
        normality_p = diagnostics.get("normality_p", np.nan)
        cards = [
            ("Residual Mean", _format_metric(diagnostics.get("mean_residual")), "chart-simple", "primary"),
            ("Residual Std", _format_metric(diagnostics.get("residual_std")), "wave-square", "success"),
            ("Bias", diagnostics.get("bias", "N/A"), "scale-balanced", "warning"),
            ("Ljung-Box p", _format_metric(lb_p), "shuffle", "danger"),
            ("Normality p", _format_metric(normality_p), "chart-area", "info"),
        ]
        autocorr_note = "Residual autocorrelation looks acceptable." if pd.notna(lb_p) and lb_p >= 0.05 else "Residual autocorrelation may remain; consider seasonality, extra lags, or another model."
        normality_note = "Residual normality is plausible." if pd.notna(normality_p) and normality_p >= 0.05 else "Residuals may be non-normal; interval quality matters more than the normality assumption."
        return ui.div(
            ui.div(
                *(
                    ui.div(
                        ui.div(ui.tags.i(class_=f"fa-solid fa-{icon}"), class_="metric-icon"),
                        ui.div(value, class_="metric-value"),
                        ui.div(label, class_="metric-label"),
                        class_=f"metric-card border-{color}",
                    )
                    for label, value, icon, color in cards
                ),
                class_="metric-grid mt-2",
            ),
            ui.div(f"{model_name}: {autocorr_note} {normality_note}", class_="alert alert-info py-2 mt-3"),
        )

    @output
    @render_widget
    def residual_diagnostics_plot():
        result = forecast_result.get()
        model_name, model_data = _best_time_series_model_data(result)
        if not model_data:
            return _empty_plotly_figure("Generate a time-series forecast to inspect residuals")

        diagnostics = model_data.get("diagnostics", {})
        residuals = np.asarray(diagnostics.get("residuals", []), dtype=float)
        residuals = residuals[np.isfinite(residuals)]
        if len(residuals) == 0:
            return _empty_plotly_figure("No finite residuals are available")

        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Residuals over time", "Residual distribution"))
        fig.add_trace(go.Scatter(x=np.arange(1, len(residuals) + 1), y=residuals, mode="lines", name="Residual"), row=1, col=1)
        fig.add_trace(go.Histogram(x=residuals, name="Residuals", marker_color="#12c6a3", opacity=0.8), row=1, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", row=1, col=1)
        _plotly_dark_layout(fig, title=f"Residual Diagnostics: {model_name}")
        fig.update_xaxes(title="Observation", row=1, col=1)
        fig.update_yaxes(title="Residual", row=1, col=1)
        fig.update_xaxes(title="Residual", row=1, col=2)
        fig.update_yaxes(title="Count", row=1, col=2)
        return fig

    @output
    @render_widget
    def backtesting_plot():
        result = forecast_result.get()
        if result is None or not result.get("ok") or result.get("kind") != "Time Series":
            return _empty_plotly_figure("Generate a time-series forecast to see validation backtests")

        models_dict = result.get("models", {})
        validation_series = [
            (name, data.get("validation_actual"), data.get("validation_predicted"))
            for name, data in models_dict.items()
            if data.get("validation_actual") is not None and data.get("validation_predicted") is not None
        ]
        if not validation_series:
            return _empty_plotly_figure("Backtest predictions are not available for this run")

        fig = go.Figure()
        first_actual = np.asarray(validation_series[0][1], dtype=float)
        x_axis = np.arange(1, len(first_actual) + 1)
        fig.add_trace(go.Scatter(x=x_axis, y=first_actual, mode="lines+markers", name="Validation Actual", line={"color": "#3487ff", "width": 2.4}))
        colors = ["#12c6a3", "#f59e0b", "#ec4899", "#8b5cf6", "#ef4444", "#3b82f6", "#22c55e"]
        for idx, (name, _, predicted) in enumerate(validation_series):
            predicted = np.asarray(predicted, dtype=float)
            fig.add_trace(
                go.Scatter(
                    x=np.arange(1, len(predicted) + 1),
                    y=predicted,
                    mode="lines",
                    name=f"{name} Backtest",
                    line={"color": colors[idx % len(colors)], "width": 1.8},
                )
            )
        _plotly_dark_layout(fig, title="Backtesting: Actual vs Predicted")
        fig.update_xaxes(title="Validation Step")
        fig.update_yaxes(title=result["response_column"])
        return fig

    @output
    @render.ui
    def anomaly_report():
        result = forecast_result.get()
        if result is None or not result.get("ok") or result.get("kind") != "Time Series":
            return ui.div("Generate a time-series forecast to detect anomalies in the selected response.", class_="alert alert-info py-2")
        anomalies = result.get("anomalies")
        count = 0 if anomalies is None else len(anomalies)
        if count == 0:
            return ui.div("No major anomalies detected by rolling-residual and IQR checks.", class_="alert alert-success py-2")
        preview = anomalies.head(5).to_dict("records")
        return ui.div(
            ui.div(f"{count:,} potential anomalies detected. Review them before deciding whether to cap, remove, or annotate them.", class_="alert alert-warning py-2"),
            ui.tags.ul(*(ui.tags.li(f"{row['Axis']}: {row['Actual']:.4g} ({row['Reason']})") for row in preview)),
        )

    @output
    @render_widget
    def anomaly_plot():
        result = forecast_result.get()
        if result is None or not result.get("ok") or result.get("kind") != "Time Series":
            return _empty_plotly_figure("Generate a time-series forecast to see anomaly markers")
        fig = go.Figure()
        actual_axis = result["actual_axis"]
        actual = np.asarray(result["actual"], dtype=float)
        fig.add_trace(go.Scatter(x=actual_axis, y=actual, mode="lines", name="Actual", line={"color": "#3487ff", "width": 2.2}))
        anomalies = result.get("anomalies")
        if anomalies is not None and len(anomalies):
            anomaly_indices = anomalies["Index"].astype(int).to_numpy() - 1
            axis_values = [actual_axis.iloc[idx] if hasattr(actual_axis, "iloc") else actual_axis[idx] for idx in anomaly_indices if 0 <= idx < len(actual)]
            y_values = [actual[idx] for idx in anomaly_indices if 0 <= idx < len(actual)]
            fig.add_trace(go.Scatter(x=axis_values, y=y_values, mode="markers", name="Potential anomaly", marker={"color": "#f59e0b", "size": 10, "symbol": "diamond"}))
        _plotly_dark_layout(fig, title="Anomaly Detection")
        fig.update_xaxes(title=result.get("x_label", "Sequence"))
        fig.update_yaxes(title=result["response_column"])
        return fig

    @output
    @render_widget
    def forecast_explainability_plot():
        result = forecast_result.get()
        model_name, model_data = _best_time_series_model_data(result)
        if not model_data:
            return _empty_plotly_figure("Generate a time-series forecast to see forecasting explainability")

        importance = model_data.get("importance")
        if importance:
            df_imp = pd.DataFrame({"Feature": importance["features"], "Importance": importance["importance"]})
            df_imp["Abs_Importance"] = df_imp["Importance"].abs()
            df_imp = df_imp.sort_values("Abs_Importance", ascending=True).tail(15)
            fig = go.Figure(go.Bar(x=df_imp["Importance"], y=df_imp["Feature"], orientation="h", marker_color="#12c6a3"))
            _plotly_dark_layout(fig, title=f"Feature Importance: {model_name}")
            return fig

        y = pd.Series(pd.to_numeric(result.get("actual", []), errors="coerce")).dropna().astype(float)
        if len(y) < 4:
            return _empty_plotly_figure("Not enough history for lag explainability")
        max_lag = min(24, len(y) // 2)
        lag_rows = []
        for lag in range(1, max_lag + 1):
            corr = y.autocorr(lag=lag)
            if pd.notna(corr):
                lag_rows.append({"Lag": f"lag_{lag}", "Autocorrelation": corr})
        if not lag_rows:
            return _empty_plotly_figure("No lag signal is available")
        lag_df = pd.DataFrame(lag_rows)
        lag_df["Abs"] = lag_df["Autocorrelation"].abs()
        lag_df = lag_df.sort_values("Abs", ascending=True).tail(15)
        fig = go.Figure(go.Bar(x=lag_df["Autocorrelation"], y=lag_df["Lag"], orientation="h", marker_color="#12c6a3"))
        _plotly_dark_layout(fig, title=f"Lag Signal: {model_name}")
        fig.update_xaxes(title="Autocorrelation")
        return fig

    @output
    @render.data_frame
    def forecast_table():
        result = forecast_result.get()
        if result is None or not result.get("ok"):
            return pd.DataFrame()
        table = result["table"].copy()
        if input.compact_forecast_table():
            best_model = result.get("best_model")
            base_columns = [column for column in ["Axis", "Sequence", "Actual"] if column in table.columns]
            model_columns = [
                column
                for column in table.columns
                if best_model and column.startswith(f"{best_model}_")
            ]
            table = table[base_columns + model_columns] if model_columns else table[base_columns or table.columns.tolist()]
        if "Axis" in table.columns:
            table["Axis"] = table["Axis"].apply(lambda value: value.strftime("%Y-%m-%d") if hasattr(value, "strftime") else value)
        return render.DataGrid(table, width="100%", height="400px", filters=True, selection_mode="none")

    @output
    @render_widget
    def explainability_plot():
        result = forecast_result.get()
        if result is None or not result.get("ok"):
            return _empty_plotly_figure("Generate a non-time-series forecast to see explainability")
            
        models_dict = result.get("models", {})
        best_model = result.get("best_model", "")
        
        importance_data = None
        if best_model in models_dict:
            importance_data = models_dict[best_model].get("importance")
            
        if not importance_data:
            return _empty_plotly_figure(f"No explainability available for {best_model}")
            
        features = importance_data["features"]
        importances = importance_data["importance"]
        
        df_imp = pd.DataFrame({"Feature": features, "Importance": importances})
        df_imp["Abs_Importance"] = df_imp["Importance"].abs()
        df_imp = df_imp.sort_values(by="Abs_Importance", ascending=True).tail(15)
        
        fig = go.Figure(go.Bar(
            x=df_imp["Importance"],
            y=df_imp["Feature"],
            orientation="h",
            marker_color="#12c6a3",
        ))
        _plotly_dark_layout(fig, title=f"Feature Importance ({best_model})")
        return fig

    @output
    @render.download(filename="forecast_results.csv")
    def download():
        result = forecast_result.get()
        if result is None or not result.get("ok"):
            yield pd.DataFrame({"Message": ["Generate a forecast before downloading results."]}).to_csv(index=False)
            return
        yield result["table"].to_csv(index=False)

    @output
    @render.download(filename="forecast_report.html")
    def download_report():
        result = forecast_result.get()
        if result is None or not result.get("ok"):
            yield "<html><body><h1>Generate a forecast before downloading report.</h1></body></html>"
            return

        df = loaded_data()
        models_dict = result.get("models", {})
        metric_keys = ["MASE", "WAPE", "sMAPE", "MAPE", "RMSE", "MAE", "R2", "BIC", "Coverage"] if result.get("metric_kind") != "classification" else ["Accuracy"]
        metric_rows = []
        for m_name, m_data in models_dict.items():
            metric_rows.append(
                "<tr>"
                f"<td>{escape(str(m_name))}</td>"
                + "".join(f"<td>{escape(_format_metric(m_data['metrics'].get(k, np.nan), '%' if k in ['MAPE', 'sMAPE', 'WAPE', 'MdAPE', 'Coverage', 'Accuracy'] else ''))}</td>" for k in metric_keys)
                + "</tr>"
            )

        quality_html = ""
        summary_html = ""
        if df is not None:
            quality = _build_data_quality_summary(df, selected_time_column())
            quality_html = (
                "<h2>Data Quality</h2><table><tr><th>Check</th><th>Count</th><th>Details</th></tr>"
                + "".join(f"<tr><td>{escape(str(issue))}</td><td>{count:,}</td><td>{escape(str(detail))}</td></tr>" for issue, count, detail in quality["rows"])
                + "</table><h3>Recommended Fixes</h3><ul>"
                + "".join(f"<li>{escape(str(item))}</li>" for item in quality["recommendations"])
                + "</ul>"
            )
            try:
                summary_html = "<h2>Summary Statistics</h2>" + df.describe(include="all", datetime_is_numeric=True).transpose().to_html()
            except TypeError:
                summary_html = "<h2>Summary Statistics</h2>" + df.describe(include="all").transpose().to_html()

        best_model = result.get("best_model")
        explainability_html = ""
        if best_model in models_dict and models_dict[best_model].get("importance"):
            importance = models_dict[best_model]["importance"]
            explainability_df = pd.DataFrame({"Feature": importance["features"], "Importance": importance["importance"]})
            explainability_df["Absolute Importance"] = explainability_df["Importance"].abs()
            explainability_html = "<h2>Explainability</h2>" + explainability_df.sort_values("Absolute Importance", ascending=False).to_html(index=False)

        scenario_html = ""
        if result.get("scenario_predictions"):
            scenario_df = pd.DataFrame(
                {
                    "Model": list(result["scenario_predictions"].keys()),
                    "Scenario Prediction": list(result["scenario_predictions"].values()),
                }
            )
            scenario_html = (
                f"<h2>Scenario Forecast</h2><p>{escape(str(result.get('scenario_feature')))} = {escape(str(result.get('scenario_value')))}</p>"
                + scenario_df.to_html(index=False)
            )

        diagnostics_html = ""
        if result.get("kind") == "Time Series":
            best = result.get("best_model")
            diagnostics = models_dict.get(best, {}).get("diagnostics", {}) if best in models_dict else {}
            if diagnostics:
                diagnostics_df = pd.DataFrame(
                    {
                        "Diagnostic": ["Residual mean", "Residual std", "Bias", "Ljung-Box p", "Normality p"],
                        "Value": [
                            _format_metric(diagnostics.get("mean_residual")),
                            _format_metric(diagnostics.get("residual_std")),
                            diagnostics.get("bias", "N/A"),
                            _format_metric(diagnostics.get("ljung_box_p")),
                            _format_metric(diagnostics.get("normality_p")),
                        ],
                    }
                )
                diagnostics_html = "<h2>Residual Diagnostics</h2>" + diagnostics_df.to_html(index=False)

        anomaly_html = ""
        if result.get("kind") == "Time Series" and result.get("anomalies") is not None:
            anomalies = result["anomalies"]
            if len(anomalies):
                anomaly_html = "<h2>Anomalies</h2>" + anomalies.to_html(index=False)
            else:
                anomaly_html = "<h2>Anomalies</h2><p>No major anomalies detected.</p>"

        table = result["table"].copy()
        if "Axis" in table.columns:
            table["Axis"] = table["Axis"].apply(lambda value: value.strftime("%Y-%m-%d") if hasattr(value, "strftime") else value)
        chart_html = _forecast_figure(result).to_html(full_html=False, include_plotlyjs="cdn")

        settings_rows = [
            ("Forecast type", result.get("kind", "")),
            ("Response variable", result.get("response_column", "")),
            ("Best model", result.get("best_model", "")),
            ("Validation", result.get("validation_method", "")),
            ("Test periods", result.get("test_periods", "")),
            ("Train split", result.get("train_split", "")),
            ("Scenario adjustment", result.get("scenario_adjustment", "")),
            ("Detected frequency", result.get("profile", {}).get("frequency_label", "")),
            ("Detected seasonal period", result.get("profile", {}).get("detected_period", "")),
        ]
        settings_html = "<table>" + "".join(f"<tr><th>{escape(str(k))}</th><td>{escape(str(v))}</td></tr>" for k, v in settings_rows if v not in (None, "")) + "</table>"

        html = f"""
        <html>
        <head>
            <title>Forecast Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; color: #172033; padding: 24px; }}
                h1, h2, h3 {{ color: #0f172a; }}
                table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }}
                th, td {{ border: 1px solid #d7dee8; padding: 8px; text-align: left; }}
                th {{ background: #eef3f8; }}
                .section {{ margin-top: 28px; }}
            </style>
        </head>
        <body>
            <h1>Forecast Report: {escape(str(result['response_column']))}</h1>
            <div class="section"><h2>Model Settings</h2>{settings_html}</div>
            <div class="section"><h2>Forecast Chart</h2>{chart_html}</div>
            <div class="section"><h2>Model Comparison</h2><table><tr><th>Model</th>{''.join(f'<th>{escape(k)}</th>' for k in metric_keys)}</tr>{''.join(metric_rows)}</table></div>
            <div class="section">{diagnostics_html}</div>
            <div class="section">{anomaly_html}</div>
            <div class="section">{quality_html}</div>
            <div class="section">{summary_html}</div>
            <div class="section">{explainability_html}</div>
            <div class="section">{scenario_html}</div>
            <div class="section"><h2>Forecast Table</h2>{table.to_html(index=False)}</div>
        </body>
        </html>
        """
        yield html
