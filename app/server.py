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

    @reactive.effect
    @reactive.event(input.upload_data, ignore_init=True)
    def _():
        ui.update_navset("main_nav", selected="summary", session=session)

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

    forecast_result = reactive.Value(None)

    def _numeric_columns(df):
        return df.select_dtypes(include="number").columns.tolist()

    def _valid_horizon():
        try:
            return max(1, int(input.horizon() or 1))
        except (TypeError, ValueError):
            return 1

    def _preferred_response_column(df, prefer_summary=False):
        numeric_columns = _numeric_columns(df)
        if not numeric_columns:
            return None

        if prefer_summary:
            for column in _normalize_selection(input.vars_stat_selected()):
                if column in numeric_columns:
                    return column

        current_response = input.response_variable()
        if current_response in numeric_columns:
            return current_response

        current_graph_y = input.y_variable_graph()
        if current_graph_y in numeric_columns:
            return current_graph_y

        return numeric_columns[0]

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

    def _regression_metrics(actual, predicted):
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        mask = np.isfinite(actual) & np.isfinite(predicted)
        if not mask.any():
            return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan}

        actual = actual[mask]
        predicted = predicted[mask]
        errors = actual - predicted
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))
        nonzero = actual != 0
        mape = float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100) if nonzero.any() else np.nan
        total = float(np.sum((actual - np.mean(actual)) ** 2))
        r2 = float(1 - np.sum(errors**2) / total) if total else np.nan
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

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
        time_column = input.time_variable()
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
        future = np.asarray(fit.forecast(steps=horizon), dtype=float)
        summary = str(fit.summary())
        if notes:
            summary = "\n".join(notes) + "\n\n" + summary
        return fitted, future, summary

    def _fit_ets(values, horizon):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        y = np.asarray(values, dtype=float)
        seasonal_period = int(input.seasonal_period() or 1) if input.seasonal() else 1
        seasonal = "add" if seasonal_period > 1 and len(y) >= seasonal_period * 2 else None
        fit = ExponentialSmoothing(
            y,
            trend="add" if len(y) >= 4 else None,
            seasonal=seasonal,
            seasonal_periods=seasonal_period if seasonal else None,
            initialization_method="estimated",
        ).fit(optimized=True)
        return np.asarray(fit.fittedvalues, dtype=float), np.asarray(fit.forecast(horizon), dtype=float), str(fit.summary())

    def _lagged_matrix(values, lags):
        y = np.asarray(values, dtype=float)
        x_rows = []
        y_rows = []
        for index in range(lags, len(y)):
            x_rows.append(y[index - lags:index])
            y_rows.append(y[index])
        return np.asarray(x_rows), np.asarray(y_rows)

    def _fit_lagged_regressor(values, horizon, model_name):
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        y = np.asarray(values, dtype=float)
        lags = max(1, min(12, len(y) // 3))
        x_train, y_train = _lagged_matrix(y, lags)
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
        future = []
        for _ in range(horizon):
            next_value = float(estimator.predict(np.asarray(history[-lags:]).reshape(1, -1))[0])
            future.append(next_value)
            history.append(next_value)

        summary = [
            f"Model: {model_name}",
            f"Lag features: {lags}",
            f"Estimator: {estimator}",
        ]
        return fitted, np.asarray(future, dtype=float), "\n".join(summary)

    def _fit_prophet(values, horizon):
        from prophet import Prophet

        y = np.asarray(values, dtype=float)
        model_df = pd.DataFrame(
            {
                "ds": pd.date_range("2000-01-01", periods=len(y), freq="D"),
                "y": y,
            }
        )
        model = Prophet()
        model.fit(model_df)
        future_frame = model.make_future_dataframe(periods=horizon, freq="D")
        predicted = model.predict(future_frame)["yhat"].to_numpy()
        summary = [
            "Model: Prophet",
            f"Changepoints: {len(model.changepoints)}",
            f"Seasonality mode: {model.seasonality_mode}",
            f"Growth: {model.growth}",
        ]
        return predicted[: len(y)], predicted[len(y):], "\n".join(summary)

    def _run_time_series_forecast(df, response_column):
        horizon = _valid_horizon()
        model_name = input.model() or "ARIMA"

        series = pd.to_numeric(df[response_column], errors="coerce")
        model_df = df.loc[series.notna()].copy().reset_index(drop=True)
        y = series.dropna().astype(float).to_numpy()
        if len(y) < 3:
            return _as_error("Select a numeric response variable with at least 3 valid observations.")

        seasonal_period = int(input.seasonal_period() or 1) if input.seasonal() else 1

        try:
            if model_name == "ETS":
                fitted, future, summary = _fit_ets(y, horizon)
            elif model_name == "Prophet":
                fitted, future, summary = _fit_prophet(y, horizon)
            elif model_name in {"GRNN", "Neural Network", "AutoML"}:
                fitted, future, summary = _fit_lagged_regressor(y, horizon, model_name)
            else:
                fitted, future, summary = _fit_arima_family(y, horizon, model_name, seasonal_period)
        except Exception as exc:
            return _as_error(f"Could not fit {model_name}: {exc}")

        actual_axis, future_axis, x_label = _future_axis_from_time(model_df, horizon)
        metrics = _regression_metrics(y, fitted)
        result_table = pd.DataFrame(
            {
                "Axis": list(actual_axis) + list(future_axis),
                "Actual": list(y) + [np.nan] * horizon,
                "Fitted": list(fitted) + [np.nan] * horizon,
                "Forecast": [np.nan] * len(y) + list(future),
            }
        )

        return {
            "ok": True,
            "kind": "Time Series",
            "model_name": model_name,
            "response_column": response_column,
            "x_label": x_label,
            "actual_axis": actual_axis,
            "future_axis": future_axis,
            "actual": y,
            "fitted": fitted,
            "future": future,
            "metrics": metrics,
            "summary": summary,
            "table": result_table,
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
        import statsmodels.api as sm

        model_name = input.model1() or "Linear Regression"
        feature_columns = _feature_columns(df, response_column)
        if not feature_columns:
            return _as_error("Select at least one predictor variable for non-time-series forecasting.")

        y_raw = df[response_column]
        y_numeric = pd.to_numeric(y_raw, errors="coerce")
        keep_rows = y_raw.notna()
        if model_name != "Logistic Regression":
            keep_rows = keep_rows & y_numeric.notna()

        model_df = df.loc[keep_rows].reset_index(drop=True)
        if model_df.empty:
            return _as_error("No complete rows are available for the selected response variable.")

        x_frame = _prepared_feature_frame(model_df, feature_columns)
        if x_frame.empty:
            return _as_error("The selected predictor variables could not be prepared for modeling.")
        x_frame = x_frame.astype(float)

        if model_name == "Logistic Regression" and y_raw.loc[keep_rows].nunique(dropna=True) <= 20:
            y = y_raw.loc[keep_rows].reset_index(drop=True)
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            model.fit(x_frame, y)
            fitted = model.predict(x_frame)
            metrics = _classification_metrics(y, fitted)
            summary = _sklearn_regression_summary(model_name, model, x_frame.columns.tolist())
            metric_kind = "classification"
        else:
            y = y_numeric.loc[keep_rows].astype(float).reset_index(drop=True)
            metric_kind = "regression"
            if model_name == "GLM":
                x_with_constant = sm.add_constant(x_frame, has_constant="add")
                model = sm.GLM(y, x_with_constant, family=sm.families.Gaussian()).fit()
                fitted = model.predict(x_with_constant)
                summary = str(model.summary())
            elif model_name == "LASSO":
                model = make_pipeline(StandardScaler(), Lasso(alpha=0.01, max_iter=10000))
                model.fit(x_frame, y)
                fitted = model.predict(x_frame)
                summary = _sklearn_regression_summary(model_name, model, x_frame.columns.tolist())
            elif model_name == "Ridge Regression":
                model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                model.fit(x_frame, y)
                fitted = model.predict(x_frame)
                summary = _sklearn_regression_summary(model_name, model, x_frame.columns.tolist())
            else:
                model = LinearRegression()
                model.fit(x_frame, y)
                fitted = model.predict(x_frame)
                if model_name == "Logistic Regression":
                    summary_note = "Logistic Regression requires a categorical response; Linear Regression was used instead.\n\n"
                else:
                    summary_note = ""
                summary = summary_note + _sklearn_regression_summary("Linear Regression", model, x_frame.columns.tolist())
            metrics = _regression_metrics(y, fitted)

        axis = np.arange(1, len(model_df) + 1)
        result_table = pd.DataFrame(
            {
                "Sequence": axis,
                "Actual": list(y),
                "Forecast": list(fitted),
            }
        )

        return {
            "ok": True,
            "kind": "Non-Time Series",
            "model_name": model_name,
            "response_column": response_column,
            "x_label": "Sequence",
            "actual_axis": axis,
            "future_axis": [],
            "actual": np.asarray(y),
            "fitted": np.asarray(fitted),
            "future": np.asarray([]),
            "metrics": metrics,
            "metric_kind": metric_kind,
            "summary": summary,
            "table": result_table,
        }

    def _build_forecast_result(response_column=None, prefer_summary_response=False):
        df = loaded_data()
        if df is None:
            return _as_error("Load data before running a forecast.")
        if df.empty:
            return _as_error("The loaded data is empty.")

        selected_response = response_column or _preferred_response_column(df, prefer_summary_response)
        if selected_response is None or selected_response not in df.columns:
            return _as_error("Select a numeric response variable before running a forecast.")

        if input.data_type() == "Non-Time Series":
            return _run_tabular_forecast(df, selected_response)
        return _run_time_series_forecast(df, selected_response)

    @reactive.effect
    @reactive.event(input.implement_forecasting, ignore_init=True)
    def _implement_forecasting_from_summary():
        df = loaded_data()
        if df is None:
            forecast_result.set(_as_error("Load data before running a forecast."))
            ui.notification_show("Load data before running a forecast.", type="error", duration=4)
            return

        response_column = _preferred_response_column(df, prefer_summary=True)
        _update_forecast_controls(df, response_column, select_all_x=True)
        ui.update_navset("main_nav", selected="forecasting", session=session)
        forecast_result.set(_build_forecast_result(response_column=response_column, prefer_summary_response=True))

    @reactive.effect
    @reactive.event(input.forecast, ignore_init=True)
    def _generate_forecast_from_forecast_tab():
        forecast_result.set(_build_forecast_result())

    def _blank_forecast_plot(message):
        fig, ax = plt.subplots(figsize=(11, 4.8))
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

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
        return ui.div(
            f"{result['model_name']} forecast generated for {result['response_column']}.",
            class_="alert alert-success py-2",
        )

    @output
    @render.plot
    def plot():
        result = forecast_result.get()
        if result is None:
            return _blank_forecast_plot("Generate a forecast to see actual vs forecast")
        if not result.get("ok"):
            return _blank_forecast_plot(result.get("message", "Forecasting failed"))

        fig, ax = plt.subplots(figsize=(11.5, 5))
        actual_axis = result["actual_axis"]
        ax.plot(actual_axis, result["actual"], label="Actual", color="#1f6feb", linewidth=2)

        fitted = np.asarray(result["fitted"])
        future = np.asarray(result["future"])
        if len(future):
            future_axis = result["future_axis"]
            ax.plot(future_axis, future, label="Forecast", color="#2da44e", linewidth=2.2, marker="o", markersize=3)
            if len(actual_axis):
                ax.axvline(actual_axis.iloc[-1] if hasattr(actual_axis, "iloc") else actual_axis[-1], color="#6b7280", linestyle="--", alpha=0.55)
        elif len(fitted):
            ax.plot(actual_axis, fitted, label="Forecast", color="#2da44e", linewidth=1.8, alpha=0.9)

        ax.set_title(f"Actual vs Forecast: {result['response_column']}", fontsize=13)
        ax.set_xlabel(result.get("x_label", "Sequence"))
        ax.set_ylabel(result["response_column"])
        ax.grid(True, alpha=0.22)
        ax.legend(loc="best")
        fig.autofmt_xdate()
        plt.tight_layout()
        return fig

    @output
    @render.ui
    def model_accuracy():
        result = forecast_result.get()
        if result is None or not result.get("ok"):
            return ui.div()

        metrics = result["metrics"]
        if result.get("metric_kind") == "classification":
            cards = [("Accuracy", _format_metric(metrics.get("Accuracy"), "%"), "success")]
        else:
            cards = [
                ("MAPE", _format_metric(metrics.get("MAPE"), "%"), "info"),
                ("RMSE", _format_metric(metrics.get("RMSE")), "success"),
                ("MAE", _format_metric(metrics.get("MAE")), "warning"),
                ("R2", _format_metric(metrics.get("R2")), "danger"),
            ]

        return ui.div(
            *(
                ui.div(
                    ui.div(value, class_="metric-value"),
                    ui.div(label, class_="metric-label"),
                    class_=f"metric-card bg-{color}",
                )
                for label, value, color in cards
            ),
            class_="metric-grid mt-3",
        )

    @output
    @render.text
    def fitted_model():
        result = forecast_result.get()
        if result is None:
            return "Generate a forecast to see fitted model parameters."
        if not result.get("ok"):
            return result.get("message", "Forecasting failed.")
        return result.get("summary", "No fitted model summary is available.")

    @output
    @render.download(filename="forecast_results.csv")
    def download():
        result = forecast_result.get()
        if result is None or not result.get("ok"):
            yield pd.DataFrame({"Message": ["Generate a forecast before downloading results."]}).to_csv(index=False)
            return
        yield result["table"].to_csv(index=False)
