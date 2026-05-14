"""
Main server logic for the PyShiny AI Forecasting Application.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny import reactive, render, req, ui

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    ARIMA = None
    ExponentialSmoothing = None
    SARIMAX = None

try:
    from prophet import Prophet
except Exception:
    Prophet = None

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression, RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
except Exception:
    RandomForestRegressor = None
    LassoCV = None
    LinearRegression = None
    LogisticRegression = None
    RidgeCV = None
    r2_score = None
    KNeighborsRegressor = None
    MLPRegressor = None
    StandardScaler = None

try:
    import requests
except Exception:
    requests = None

try:
    import seaborn as sns
except Exception:
    sns = None


APP_DIR = Path(__file__).resolve().parents[1]
SAMPLE_FILE = APP_DIR / "timeseries_demo.csv"


def _empty_plot(message: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    return fig


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def _first_column_series(df: pd.DataFrame, column: str) -> pd.Series:
    values = df.loc[:, column]
    if isinstance(values, pd.DataFrame):
        return values.iloc[:, 0]
    return values


def _selected_columns(value: Any, available: list[str]) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if value == "":
            return []
        selected = [value]
    else:
        selected = list(value)
    return [col for col in selected if col in available]


def _validate_data(df: pd.DataFrame) -> tuple[bool, str]:
    if df.empty:
        return False, "The file is empty."
    numeric_count = len(_numeric_columns(df))
    if numeric_count < max(1, df.shape[1] - 1):
        return False, "Invalid file format. Please download the template."
    return True, "Successfully Uploaded."


def _clean_data(df: pd.DataFrame, time_var: str | None) -> pd.DataFrame:
    clean = df.copy()
    if time_var in clean.columns:
        parsed = pd.to_datetime(clean[time_var], errors="coerce")
        sort_col = parsed if parsed.notna().all() else clean[time_var]
        clean = clean.assign(**{time_var: sort_col}).sort_values(time_var, kind="mergesort")

    for col in _numeric_columns(clean):
        series = pd.to_numeric(clean[col], errors="coerce")
        median = series.median()
        if pd.isna(median):
            median = 0
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if pd.notna(iqr) and iqr > 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            series = series.mask((series < lower) | (series > upper), median)
        clean[col] = series.fillna(median)

    return clean.reset_index(drop=True)


def _summary_stats(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    if columns:
        selected = _selected_columns(columns, numeric.columns.tolist())
        numeric = numeric[selected]
    if numeric.empty:
        return pd.DataFrame({"Note": ["No numeric columns found"]})

    rows: list[dict[str, Any]] = []
    calculations = [
        ("Sum", lambda x: x.sum(skipna=True)),
        ("Observations", lambda x: x.sum(skipna=True) / x.mean(skipna=True)),
        ("Mean", lambda x: x.mean(skipna=True)),
        ("Median", lambda x: x.median(skipna=True)),
        ("Mode", lambda x: x.mode(dropna=True).iloc[0] if not x.mode(dropna=True).empty else np.nan),
        ("Standard Deviation", lambda x: x.std(skipna=True)),
        ("Variance", lambda x: x.var(skipna=True)),
        ("Maximum", lambda x: x.max(skipna=True)),
        ("Minimum", lambda x: x.min(skipna=True)),
        ("Skewness", lambda x: x.skew(skipna=True)),
        ("Kurtosis", lambda x: x.kurt(skipna=True)),
        ("Interquartile Range", lambda x: x.quantile(0.75) - x.quantile(0.25)),
    ]
    for label, func in calculations:
        row = {"type": label}
        for col in numeric.columns:
            value = func(_first_column_series(numeric, col))
            row[col] = round(value, 4) if pd.notna(value) and np.isscalar(value) else value
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_table_for_display(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    for col in table.columns:
        if col != "type" and pd.api.types.is_numeric_dtype(table[col]):
            table[col] = pd.to_numeric(table[col], errors="coerce").round(4)
    return table


def _summary_table_ui(df: pd.DataFrame | None):
    if df is None or df.empty:
        return ui.div("Prepare data to see summary statistics.", class_="alert alert-info")
    table = _summary_table_for_display(df)
    return ui.HTML(
        table.to_html(
            index=False,
            classes="table table-striped table-hover table-sm summary-table",
            border=0,
            na_rep="",
        )
    )


def _summary_for_selection(df: pd.DataFrame | None, selected: Any = None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame({"Note": ["Prepare data to see summary statistics"]})
    numeric_cols = _numeric_columns(df)
    selected_cols = _selected_columns(selected, numeric_cols)
    return _summary_stats(df, selected_cols)


def _summary_numeric_for_selection(df: pd.DataFrame | None, selected: Any = None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    numeric_cols = _numeric_columns(df)
    selected_cols = _selected_columns(selected, numeric_cols)
    active_cols = selected_cols or numeric_cols
    if not active_cols:
        return pd.DataFrame()
    numeric = df[active_cols].apply(pd.to_numeric, errors="coerce")
    return numeric.dropna(axis=1, how="all")


def _summary_plot(numeric: pd.DataFrame | None, plot_type: str):
    if numeric is None or numeric.empty:
        return _empty_plot("No numeric columns selected")

    series_list = [numeric[col].dropna().to_numpy() for col in numeric.columns]
    series_list = [values for values in series_list if len(values) > 0]
    if not series_list:
        return _empty_plot("Selected columns contain no numeric values")

    labels = [col for col in numeric.columns if len(numeric[col].dropna()) > 0]
    if plot_type == "Histogram":
        rows = int(np.ceil(len(labels) / 2))
        fig, axes = plt.subplots(rows, min(2, len(labels)), figsize=(11, max(4, rows * 3)))
        axes = np.atleast_1d(axes).ravel()
        for idx, col in enumerate(labels):
            axes[idx].hist(numeric[col].dropna(), bins=20, color="#026efa", alpha=0.75)
            axes[idx].set_title(col)
            axes[idx].tick_params(axis="x", labelrotation=30)
        for extra in axes[len(labels) :]:
            extra.set_visible(False)
    else:
        fig, ax = plt.subplots(figsize=(11, 5))
        if plot_type == "Boxplot":
            ax.boxplot(series_list, labels=labels, patch_artist=True)
            ax.set_title("Boxplot")
        else:
            ax.violinplot(series_list, showmeans=True)
            ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
            ax.set_title("Violin Plot")
        ax.tick_params(axis="x", labelrotation=30)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    return fig


def _metrics(actual: np.ndarray, fitted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(fitted)
    actual = actual[mask]
    fitted = fitted[mask]
    non_zero = actual != 0
    if len(actual) == 0:
        return {"MAPE": np.nan, "RMSPE": np.nan, "R2 Score": np.nan}
    mape = np.mean(np.abs((actual[non_zero] - fitted[non_zero]) / actual[non_zero])) if non_zero.any() else np.nan
    rmspe = np.sqrt(np.mean(((actual[non_zero] - fitted[non_zero]) / actual[non_zero]) ** 2)) if non_zero.any() else np.nan
    if r2_score is not None and len(actual) > 1:
        r2 = r2_score(actual, fitted)
    else:
        ss_res = np.sum((actual - fitted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
    return {"MAPE": mape, "RMSPE": rmspe, "R2 Score": r2}


def _metric_color(metric: str, value: float) -> str:
    if not np.isfinite(value):
        return "secondary"
    if metric == "R2 Score":
        return "success" if value > 0.85 else "warning" if value > 0.70 else "danger"
    return "success" if value < 0.10 else "warning" if value < 0.20 else "danger"


def _future_x(df: pd.DataFrame, time_var: str | None, horizon: int) -> np.ndarray | pd.DatetimeIndex:
    if time_var and time_var in df.columns:
        dates = pd.to_datetime(df[time_var], errors="coerce")
        if dates.notna().all() and len(dates) > 1:
            freq = pd.infer_freq(dates)
            offset = pd.tseries.frequencies.to_offset(freq) if freq else dates.diff().dropna().median()
            return pd.date_range(dates.iloc[-1] + offset, periods=horizon, freq=offset)
    return np.arange(len(df) + 1, len(df) + horizon + 1)


def _forecast_time_series(
    y: pd.Series,
    horizon: int,
    model_name: str,
    seasonal: bool,
    seasonal_period: int | None,
    dates: pd.Series | None,
) -> dict[str, Any]:
    values = pd.to_numeric(y, errors="coerce").astype(float)
    values = values.fillna(values.median() if values.notna().any() else 0)
    period = int(seasonal_period or 12)

    if model_name == "Prophet":
        if Prophet is None:
            raise RuntimeError("Prophet is not installed. Install prophet or choose another model.")
        ds = pd.to_datetime(dates, errors="coerce") if dates is not None else pd.date_range("2000-01-01", periods=len(values))
        prophet_df = pd.DataFrame({"ds": ds, "y": values})
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.fit(prophet_df)
        freq = pd.infer_freq(prophet_df["ds"]) or "D"
        future = model.make_future_dataframe(periods=horizon, freq=freq)
        pred = model.predict(future)
        fitted = pred["yhat"].iloc[: len(values)].to_numpy()
        forecast = pred["yhat"].iloc[len(values) :].to_numpy()
        summary = "Prophet model fitted with yearly and weekly seasonality."
        return {"fitted": fitted, "forecast": forecast, "model": model, "summary": summary}

    try:
        if model_name == "SARIMA":
            model = SARIMAX(values, order=(1, 1, 1), seasonal_order=(1, 1, 1, period), enforce_stationarity=False)
            fit = model.fit(disp=False)
        elif model_name == "State Space ARIMA":
            model = SARIMAX(values, order=(1, 1, 1), trend="c", enforce_stationarity=False)
            fit = model.fit(disp=False)
        elif model_name in {"ETS", "ARCH", "GARCH"}:
            trend = "add" if len(values) >= 4 else None
            seasonal_arg = "add" if seasonal and len(values) >= period * 2 else None
            fit = ExponentialSmoothing(values, trend=trend, seasonal=seasonal_arg, seasonal_periods=period if seasonal_arg else None).fit()
        else:
            fit = ARIMA(values, order=(1, 1, 1)).fit()
        fitted = np.asarray(fit.fittedvalues)
        forecast = np.asarray(fit.forecast(horizon))
        summary = str(fit.summary()) if hasattr(fit, "summary") else repr(fit)
        return {"fitted": fitted, "forecast": forecast, "model": fit, "summary": summary}
    except Exception:
        pass

    x = np.arange(len(values)).reshape(-1, 1)
    future = np.arange(len(values), len(values) + horizon).reshape(-1, 1)
    if model_name == "GRNN" and KNeighborsRegressor is not None:
        reg = KNeighborsRegressor(n_neighbors=min(5, len(values)), weights="distance")
    elif model_name == "Neural Network" and MLPRegressor is not None:
        reg = MLPRegressor(hidden_layer_sizes=(32, 16), random_state=42, max_iter=1000)
    elif model_name == "AutoML" and RandomForestRegressor is not None:
        reg = RandomForestRegressor(n_estimators=150, random_state=42)
    else:
        reg = LinearRegression() if LinearRegression is not None else None
    if reg is None:
        fitted = np.repeat(values.iloc[-1], len(values))
        forecast = np.repeat(values.iloc[-1], horizon)
        return {"fitted": fitted, "forecast": forecast, "model": None, "summary": "Naive last-value forecast."}
    reg.fit(x, values)
    return {
        "fitted": np.asarray(reg.predict(x)),
        "forecast": np.asarray(reg.predict(future)),
        "model": reg,
        "summary": f"{model_name} fitted using {reg.__class__.__name__}.",
    }


def _forecast_non_time_series(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    model_name: str,
) -> dict[str, Any]:
    model_df = df[[y_col] + x_cols].copy()
    for col in model_df.columns:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    model_df = model_df.dropna()
    if model_df.empty:
        raise RuntimeError("Selected model columns do not contain usable numeric data.")

    x = model_df[x_cols].to_numpy()
    y = model_df[y_col].to_numpy()
    scaler = StandardScaler() if StandardScaler is not None else None
    x_fit = scaler.fit_transform(x) if scaler is not None else x

    if model_name == "LASSO" and LassoCV is not None:
        model = LassoCV(cv=min(5, len(model_df)), random_state=42).fit(x_fit, y)
        fitted = model.predict(x_fit)
    elif model_name == "Ridge Regression" and RidgeCV is not None:
        model = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(x_fit, y)
        fitted = model.predict(x_fit)
    elif model_name == "Logistic Regression" and LogisticRegression is not None and len(np.unique(y)) <= 2:
        model = LogisticRegression(max_iter=1000).fit(x_fit, y)
        fitted = model.predict_proba(x_fit)[:, -1]
    else:
        model = LinearRegression().fit(x_fit, y) if LinearRegression is not None else None
        if model is None:
            fitted = np.repeat(np.mean(y), len(y))
        else:
            fitted = model.predict(x_fit)

    coef = getattr(model, "coef_", "not available")
    intercept = getattr(model, "intercept_", "not available")
    summary = f"{model_name}\nRows used: {len(model_df)}\nFeatures: {', '.join(x_cols)}\nIntercept: {intercept}\nCoefficients: {coef}"
    return {"fitted": np.asarray(fitted), "forecast": np.asarray(fitted), "model": model, "summary": summary, "actual": y}


def _chat_response(prompt: str, model_name: str, temperature: float) -> str:
    if model_name == "gpt-3.5-turbo" and requests is not None and os.getenv("OPENAI_API_KEY"):
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            },
            timeout=60,
        )
        if response.ok:
            return response.json()["choices"][0]["message"]["content"]
        return f"OpenAI API error: {response.status_code} {response.text[:300]}"
    return (
        "No configured API key is available for this model in the PyShiny app. "
        "Set the appropriate environment variable and resend. Your prompt was:\n\n"
        f"{prompt}"
    )


def _input_value(input, name: str, default=None):
    try:
        return getattr(input, name)()
    except Exception:
        return default


def server_function(input, output, session):
    chat_log = reactive.Value([])

    @reactive.calc
    @reactive.event(input.file, input.load_mongo, ignore_init=False)
    def raw_data_calc():
        source = input.data_source()
        if source == "Upload":
            file_info = input.file()
            if not file_info:
                return None
            try:
                return pd.read_csv(file_info[0]["datapath"])
            except Exception:
                return None
        else:
            if input.load_mongo() == 0:
                return None
            try:
                return pd.read_csv(SAMPLE_FILE)
            except Exception:
                return None

    @reactive.calc
    @reactive.event(input.upload_data, ignore_init=False)
    def prepared_data_calc():
        if input.upload_data() == 0:
            return None
        df = raw_data_calc()
        if df is None or df.empty:
            return None
        time_var = _input_value(input, "time_variable")
        return _clean_data(df.copy(), time_var)

    @reactive.calc
    @reactive.event(input.implement_forecasting, ignore_init=False)
    def forecast_data_calc():
        if input.implement_forecasting() == 0:
            return None
        return prepared_data_calc()

    @reactive.effect
    def _update_raw_choices():
        df = raw_data_calc()
        if df is None:
            return
        cols = df.columns.tolist()
        numeric = _numeric_columns(df)
        selected_time = cols[0] if cols else None
        selected_y = numeric[0] if numeric else selected_time
        selected_y = _input_value(input, "y_variable_graph", selected_y) or selected_y
        if selected_y not in cols:
            selected_y = numeric[0] if numeric else selected_time
        x_choices = [col for col in cols if col != selected_y] or cols
        ui.update_select("time_variable", choices=cols, selected=selected_time, session=session)
        ui.update_select("y_variable_graph", choices=cols, selected=selected_y, session=session)
        ui.update_select("x_variables_graph", choices=x_choices, selected=x_choices[0] if x_choices else None, session=session)

    @reactive.effect
    def _update_prepared_choices():
        df = prepared_data_calc()
        if df is None:
            return
        cols = df.columns.tolist()
        numeric = _numeric_columns(df)
        selected_y = numeric[0] if numeric else (cols[0] if cols else None)
        with reactive.isolate():
            current_y = _input_value(input, "response_variable", selected_y)
        if current_y in cols:
            selected_y = current_y
        x_choices = [col for col in numeric if col != selected_y]
        with reactive.isolate():
            current_selected = _selected_columns(_input_value(input, "vars_stat_selected", []), numeric)
        ui.update_selectize("vars_stat_selected", choices=numeric, selected=current_selected or numeric, session=session)
        ui.update_select("response_variable", choices=cols, selected=selected_y, session=session)
        ui.update_selectize(
            "x_variables",
            choices=x_choices,
            selected=x_choices[:1],
            session=session,
        )

    @reactive.effect
    def _update_x_variables_for_response():
        df = prepared_data_calc()
        if df is None:
            return
        selected_y = _input_value(input, "response_variable")
        numeric = _numeric_columns(df)
        x_choices = [col for col in numeric if col != selected_y]
        with reactive.isolate():
            current_x = _input_value(input, "x_variables", []) or []
        if isinstance(current_x, str):
            current_x = [current_x]
        selected = [col for col in current_x if col in x_choices] or x_choices[:1]
        ui.update_selectize("x_variables", choices=x_choices, selected=selected, session=session)

    @output
    @render.ui
    def file_feedback():
        df = raw_data_calc()
        if df is None:
            return ui.div("Upload a CSV or load the sample data.", class_="alert alert-info py-2")
        valid, message = _validate_data(df)
        status = "success" if valid else "danger"
        return ui.div(message, class_=f"alert alert-{status} py-2")

    @output
    @render.ui
    def info_data():
        df = raw_data_calc()
        if df is None:
            return ui.card(ui.card_header("Data Info"), ui.p("No data loaded yet."))
        missing_pct = round((df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 1) if df.size else 0
        cards = [
            ("Observations", f"{df.shape[0]:,}", "tower-observation", "warning"),
            ("Variables", f"{df.shape[1]:,}", "square-root-variable", "success"),
            ("Missing %", f"{missing_pct}%", "percentage", "danger"),
            ("Numeric Columns", f"{len(_numeric_columns(df)):,}", "list-ol", "info"),
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

    @reactive.effect
    @reactive.event(input.upload_data)
    def _move_to_summary():
        df = raw_data_calc()
        if df is not None and not df.empty:
            ui.update_navset("main_nav", selected="summary", session=session)

    @output
    @render.plot
    def vis_data():
        df = raw_data_calc()
        if df is None:
            return _empty_plot("Upload or load data to see visualization")
        x_col = _input_value(input, "x_variables_graph")
        y_col = _input_value(input, "y_variable_graph")
        if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
            return _empty_plot("Select valid variables")
        y_values = pd.to_numeric(_first_column_series(df, y_col), errors="coerce")
        if x_col == y_col:
            x_values = np.arange(1, len(df) + 1)
            x_label = "Sequence"
        else:
            x_values = _first_column_series(df, x_col)
            parsed_x = pd.Series([pd.NaT] * len(x_values))
            if not pd.api.types.is_numeric_dtype(x_values):
                parsed_x = pd.to_datetime(x_values, errors="coerce")
            if parsed_x.notna().all():
                x_values = parsed_x
            x_label = x_col
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(x_values, y_values, marker="o", linewidth=1.8, color="#026efa")
        ax.set_title(f"{y_col} by {x_label}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.25)
        fig.autofmt_xdate()
        plt.tight_layout()
        return fig

    @output
    @render.ui
    def summary_status():
        df = prepared_data_calc()
        if df is None:
            return ui.div("Step 2: prepare data from the Data tab to populate this page.", class_="alert alert-info")
        return ui.div(
            f"Step 2: using prepared data with {df.shape[0]:,} rows and {df.shape[1]:,} columns.",
            class_="alert alert-success",
        )

    @output
    @render.data_frame
    def summary_stat_table():
        try:
            df = prepared_data_calc()
            if df is None:
                return pd.DataFrame({"Note": ["Prepare data to see summary statistics."]})
            selected = _input_value(input, "vars_stat_selected", [])
            summary_df = _summary_for_selection(df, selected)
            return render.DataGrid(summary_df, width="100%", height="auto")
        except Exception as exc:
            return pd.DataFrame({"Error": [f"Unable to render summary statistics: {exc}"]})

    @output
    @render.plot
    def summary_stat_vis():
        try:
            df = prepared_data_calc()
            if df is None:
                return _empty_plot("Prepare data to see visualization")
            selected = _input_value(input, "vars_stat_selected", [])
            numeric_df = _summary_numeric_for_selection(df, selected)
            plot_type = _input_value(input, "summary_stat_plot_type", "Violin Plot")
            return _summary_plot(numeric_df, plot_type)
        except Exception as exc:
            return _empty_plot(f"Unable to render summary visualization: {exc}")

    @reactive.effect
    @reactive.event(input.implement_forecasting)
    def _go_to_forecasting():
        df = prepared_data_calc()
        if df is None or df.empty:
            ui.update_navset("main_nav", selected="data", session=session)
            return
        ui.update_navset("main_nav", selected="forecasting", session=session)

    @reactive.calc
    @reactive.event(input.forecast, ignore_init=False)
    def forecast_result_calc():
        if input.forecast() == 0:
            return None
        df = forecast_data_calc()
        if df is None or df.empty:
            return {"error": "Prepare data from the Data tab before generating a forecast."}
        try:
            y_col = input.response_variable()
            if not y_col or y_col not in df.columns:
                return {"error": "Select a valid response variable."}
            horizon = max(1, int(input.horizon() or 1))

            if input.data_type() == "Time Series":
                time_var = _input_value(input, "time_variable")
                dates = df[time_var] if time_var in df.columns else None
                x_actual = np.arange(1, len(df) + 1)
                if dates is not None:
                    parsed_dates = pd.to_datetime(dates, errors="coerce")
                    x_actual = parsed_dates if parsed_dates.notna().all() else dates
                model_name = input.model()
                result = _forecast_time_series(
                    _first_column_series(df, y_col),
                    horizon,
                    model_name,
                    bool(input.seasonal()),
                    int(input.seasonal_period() or 12) if bool(input.seasonal()) else None,
                    dates,
                )
                result.update(
                    {
                        "type": "Time Series",
                        "model_name": model_name,
                        "actual": pd.to_numeric(_first_column_series(df, y_col), errors="coerce").to_numpy(),
                        "x_actual": x_actual,
                        "x_forecast": _future_x(df, time_var, horizon),
                        "target": y_col,
                    }
                )
            else:
                x_cols = input.x_variables() or []
                if isinstance(x_cols, str):
                    x_cols = [x_cols]
                if not x_cols:
                    return {"error": "Select at least one X variable for non-time-series forecasting."}
                model_name = input.model1()
                result = _forecast_non_time_series(df, y_col, x_cols, model_name)
                result.update(
                    {
                        "type": "Non-Time Series",
                        "model_name": model_name,
                        "x_actual": np.arange(1, len(result["actual"]) + 1),
                        "x_forecast": np.arange(1, len(result["forecast"]) + 1),
                        "target": y_col,
                    }
                )
            return result
        except Exception as exc:
            return {"error": str(exc), "summary": f"Forecast failed: {exc}"}

    @output
    @render.ui
    def forecast_status():
        df = forecast_data_calc()
        if df is None:
            return ui.div("Step 3: no prepared data available. Go to Data and click Upload data.", class_="alert alert-info")
        result = forecast_result_calc()
        if result is None:
            return ui.div(
                f"Step 3: prepared data loaded with {df.shape[0]:,} rows. Choose model settings and click Generate Forecast.",
                class_="alert alert-success",
            )
        if "error" in result:
            return ui.div(result["error"], class_="alert alert-danger")
        return ui.div(f"Forecast complete using {result.get('model_name', 'selected model')}.", class_="alert alert-success")

    @output
    @render.plot
    def plot():
        result = forecast_result_calc()
        if result is None:
            return _empty_plot("Prepare data and click Generate Forecast")
        if "error" in result:
            return _empty_plot(result["error"])
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(result["x_actual"], result["actual"], label="Data", color="#026efa", linewidth=2)
        if result["type"] == "Time Series":
            ax.plot(result["x_actual"], result["fitted"], label="Fitted", color="#40a2ed", linewidth=1.5, alpha=0.75)
            ax.plot(result["x_forecast"], result["forecast"], label="Forecast", color="#e63946", linewidth=2)
        else:
            ax.plot(result["x_forecast"], result["forecast"], label="Forecast", color="#e63946", linewidth=2)
        ax.set_title("Actual Data and Forecast")
        ax.set_xlabel("Time")
        ax.set_ylabel(result["target"])
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.autofmt_xdate()
        plt.tight_layout()
        return fig

    @output
    @render.text
    def fitted_model():
        result = forecast_result_calc()
        if result is None:
            return "Generate a forecast to see the fitted model summary."
        return result.get("summary", result.get("error", "No model summary available."))

    @output
    @render.ui
    def model_accuracy():
        result = forecast_result_calc()
        if result is None or "error" in result:
            return ui.div()
        fitted = result["fitted"][: len(result["actual"])] if result["type"] == "Time Series" else result["forecast"]
        metric_values = _metrics(result["actual"][: len(fitted)], fitted)
        return ui.div(
            *(
                ui.div(
                    ui.div(f"{value * 100:.1f}%" if np.isfinite(value) else "NA", class_="metric-value"),
                    ui.div(label, class_="metric-label"),
                    class_=f"metric-card bg-{_metric_color(label, value)}",
                )
                for label, value in metric_values.items()
            ),
            class_="metric-grid",
        )

    @output
    @render.download(filename=lambda: f"forecast_{(forecast_result_calc() or {}).get('model_name', 'model')}.csv")
    def download():
        result = forecast_result_calc()
        if result is None or "error" in result:
            yield "No forecast available"
            return
        if result["type"] == "Time Series":
            forecast_df = pd.DataFrame(
                {
                    "Sequence": np.arange(1, len(result["actual"]) + len(result["forecast"]) + 1),
                    "Actuals": np.concatenate([result["actual"], np.repeat(np.nan, len(result["forecast"]))]),
                    "Forecast": np.concatenate([result["fitted"], result["forecast"]]),
                }
            )
        else:
            forecast_df = pd.DataFrame({"Sequence": result["x_forecast"], "Actuals": result["actual"], "Forecast": result["forecast"]})
        yield forecast_df.to_csv(index=False)

    @output
    @render.download(filename="summary_stat.csv")
    def summary_stat_download():
        df = prepared_data_calc()
        selected = _input_value(input, "vars_stat_selected", [])
        summary_df = _summary_for_selection(df, selected)
        yield summary_df.to_csv(index=False)

    @output
    @render.download(filename="Sample data.csv")
    def file_template_download():
        df = pd.read_csv(SAMPLE_FILE)
        yield df.to_csv(index=False)

    @reactive.effect
    @reactive.event(input.chat)
    def _send_chat():
        prompt = (input.prompt() or "").strip()
        if not prompt:
            return
        file_info = input.file_chat()
        if file_info:
            ext = Path(file_info[0]["name"]).suffix.lower()
            if ext not in {".docx", ".pptx"}:
                ui.modal_show(ui.modal("Uploaded file type should be .docx or .pptx.", title="Uploaded file type error"))
                return
            prompt = f"{prompt}\n\nUploaded file context: {file_info[0]['name']}"
        history = chat_log.get()
        response = _chat_response(prompt, input.model_gen(), float(input.temperature()))
        chat_log.set(history + [{"role": "User", "content": prompt}, {"role": "Assistant", "content": response}])
        ui.update_text_area("prompt", value="", session=session)

    @reactive.effect
    @reactive.event(input.remove_chatThread)
    def _clear_chat():
        chat_log.set([])
        ui.update_text_area("prompt", value="", session=session)

    @output
    @render.ui
    def chat_history():
        history = chat_log.get()
        if not history:
            return ui.p("No chat messages yet.", class_="text-muted")
        return ui.div(
            *(
                ui.div(
                    ui.div(item["role"], class_="chat-role"),
                    ui.markdown(item["content"]),
                    class_=f"chat-message {'chat-user' if item['role'] == 'User' else 'chat-assistant'}",
                )
                for item in history
            )
        )

    @output
    @render.download(filename="chat_history.txt")
    def download_chat():
        text = "\n\n".join(f"{item['role']}:\n{item['content']}" for item in chat_log.get())
        yield text or "No chat messages yet."
