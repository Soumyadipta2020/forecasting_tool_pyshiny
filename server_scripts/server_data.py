"""
Data handling and visualization logic for the AI Forecasting Application.
Contains file upload, data preview, visualization, and summary stats functions.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

def handle_file_upload(input, session, data):
    from shiny import reactive
    @reactive.effect
    def _():
        file_info = input.file()
        if file_info and len(file_info) > 0:
            file_path = file_info[0]["datapath"]
            df = pd.read_csv(file_path)
            data.set(df)
            session.ui.update_select(
                "time_variable",
                choices=df.columns.tolist(),
                selected=df.columns[0] if len(df.columns) > 0 else None
            )
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            session.ui.update_select(
                "target_variable",
                choices=numeric_cols,
                selected=numeric_cols[0] if len(numeric_cols) > 0 else None
            )

def render_uploaded_data(output, data):
    from shiny import render
    @output
    @render.data_frame
    def uploaded_data():
        if data.get() is not None:
            return data.get()
        return pd.DataFrame()

def render_data_viz(output, data, input):
    from shiny import render
    @output
    @render.plot
    def data_viz():
        if data.get() is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Upload data to see visualization", ha='center', va='center', transform=ax.transAxes)
            return fig
        df = data.get()
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            target_col = numeric_cols[0]
            fig, ax = plt.subplots(figsize=(10, 6))
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                date_col = date_cols[0]
                if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                ax.plot(df[date_col], df[target_col], marker='o', linestyle='-', alpha=0.7)
                ax.set_xlabel(date_col)
            else:
                ax.plot(df.index, df[target_col], marker='o', linestyle='-', alpha=0.7)
                ax.set_xlabel('Index')
            ax.set_ylabel(target_col)
            ax.set_title(f'Time Series Plot of {target_col}')
            ax.grid(True, alpha=0.3)
            try:
                import numpy as np
                z = np.polyfit(range(len(df)), df[target_col], 1)
                p = np.poly1d(z)
                ax.plot(range(len(df)), p(range(len(df))), "r--", alpha=0.7, label='Trend')
                ax.legend()
            except:
                pass
            plt.tight_layout()
            return fig
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No numeric columns found for visualization", ha='center', va='center', transform=ax.transAxes)
            return fig

def render_summary_stats(output, data):
    from shiny import render
    @output
    @render.table
    def summary_stats():
        if data.get() is None:
            return pd.DataFrame({'Note': ['Upload data to see summary statistics']})
        df = data.get()
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            return pd.DataFrame({'Note': ['No numeric columns found in the data']})
        stats = numeric_df.describe().T
        stats['skew'] = numeric_df.skew()
        stats['kurtosis'] = numeric_df.kurtosis()
        stats = stats.round(2)
        stats = stats.reset_index().rename(columns={'index': 'variable'})
        return stats

def render_stats_viz(output, data, input):
    from shiny import render
    @output
    @render.plot
    def stats_viz():
        if data.get() is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Upload data to see visualization", ha='center', va='center', transform=ax.transAxes)
            return fig
        df = data.get()
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No numeric columns found for visualization", ha='center', va='center', transform=ax.transAxes)
            return fig
        plot_type = input.plot_type()
        fig, ax = plt.subplots(figsize=(10, 6))
        if plot_type == "Boxplot":
            sns.boxplot(data=numeric_df, ax=ax)
            ax.set_title("Boxplot of Numeric Variables")
            ax.set_xlabel("Variables")
            ax.set_ylabel("Values")
            plt.xticks(rotation=45)
        elif plot_type == "Violin Plot":
            sns.violinplot(data=numeric_df, ax=ax)
            ax.set_title("Violin Plot of Numeric Variables")
            ax.set_xlabel("Variables")
            ax.set_ylabel("Values")
            plt.xticks(rotation=45)
        elif plot_type == "Histogram":
            plt.close(fig)
            n_cols = len(numeric_df.columns)
            n_rows = (n_cols + 1) // 2
            fig, axes = plt.subplots(n_rows, min(n_cols, 2), figsize=(12, 3*n_rows))
            axes = axes.flatten() if n_cols > 1 else [axes]
            for i, col in enumerate(numeric_df.columns):
                if i < len(axes):
                    sns.histplot(numeric_df[col], kde=True, ax=axes[i])
                    axes[i].set_title(f"Histogram of {col}")
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
        plt.tight_layout()
        return fig

def render_download_summary_stats(output, data):
    from shiny import render
    @output
    @render.download(filename="summary_statistics.csv")
    def download_summary_stats():
        if data.get() is None:
            return "No data available"
        df = data.get()
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            return "No numeric data available"
        stats = numeric_df.describe().T
        stats['skew'] = numeric_df.skew()
        stats['kurtosis'] = numeric_df.kurtosis()
        stats = stats.round(2)
        stats = stats.reset_index().rename(columns={'index': 'variable'})
        return stats.to_csv(index=False)

def render_download_template(output):
    from shiny import render
    @output
    @render.download(filename="sample_data.csv")
    def download_template():
        sample_data = pd.read_csv("timeseries_demo.csv")
        buffer = StringIO()
        sample_data.to_csv(buffer, index=False)
        buffer.seek(0)
        yield buffer.getvalue()
