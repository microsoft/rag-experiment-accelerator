import ast

import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


def generate_metrics(experiment_name, run_id, mlflow_client):
    """
    Generates metrics for a given experiment and run ID.

    Args:
        experiment_name (str): The name of the experiment.
        run_id (int): The ID of the run.
        mlflow_client (mlflow.MlflowClient): The MLflow client to use for logging the metrics.

    Returns:
        None
    """
    experiment = dict(mlflow_client.get_experiment_by_name(experiment_name))
    runs_list = mlflow_client.search_runs([experiment["experiment_id"]])

    models_metrics = {}
    metrics_to_plot = []
    runs_id_to_plot = []

    if len(runs_list) > 0:
        for run in runs_list:
            run_dict = run.to_dictionary()
            single_run_id = run_dict["info"]["run_id"]
            runs_id_to_plot.append(single_run_id)
            if run.data.params.get("run_metrics", {}) != {}:
                metrics = ast.literal_eval(run.data.params["run_metrics"])

                for metric_type, metric_value in metrics.items():
                    if models_metrics.get(metric_type, {}) == {}:
                        metrics_to_plot.append(metric_type)
                        models_metrics[metric_type] = {}

                    models_metrics[metric_type][single_run_id] = metric_value
                logger.debug(models_metrics)
    else:
        current_run = mlflow_client.get_run(run_id)
        if current_run.data.params.get("run_metrics", {}) != {}:
            metrics = ast.literal_eval(current_run.data.params["run_metrics"])
            for metric_type, metric_value in metrics.items():
                if models_metrics.get(metric_type, {}) == {}:
                    metrics_to_plot.append(metric_type)
                    models_metrics[metric_type] = {}

                models_metrics[metric_type][run_id] = metric_value

    x_axis = []
    y_axis = []

    fig = go.Figure()

    for metric in metrics_to_plot:
        for key, value in models_metrics[metric].items():
            x_axis.append(key)
            y_axis.append(value)

        label = key
        px.line(x_axis, y_axis)
        fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode="lines+markers", name=label))

        fig.update_layout(
            xaxis_title="run name", yaxis_title=metric, font=dict(size=15)
        )

        plot_name = metric + ".html"
        mlflow_client.log_figure(run_id, fig, plot_name)

        fig.data = []
        fig.layout = {}
        x_axis = []
        y_axis = []


def draw_hist_df(df, run_id, mlflow_client):
    """
    Draw a histogram of the given dataframe and log it to the specified run ID.

    Args:
        df (pandas.DataFrame): The dataframe to draw the histogram from.
        run_id (str): The ID of the run to log the histogram to.
        mlflow_client (mlflow.MlflowClient): The MLflow client to use for logging the histogram.

    Returns:
        None
    """
    fig = px.bar(
        x=df.columns,
        y=df.values.tolist(),
        title="metric comparison",
        color=df.columns,
        labels=dict(x="Metric Type", y="Score", color="Metric Type"),
    )
    plot_name = "all_metrics_current_run.html"
    mlflow_client.log_figure(run_id, fig, plot_name)


def plot_apk_scores(df, run_id, mlflow_client):
    fig = px.line(df, x="k", y="score", title="AP@k scores", color="search_type")
    plot_name = "average_precision_at_k.html"
    mlflow_client.log_figure(run_id, fig, plot_name)


# maybe pull these 2 above and below functions into a single one
def plot_mapk_scores(df, run_id, mlflow_client):
    fig = px.line(df, x="k", y="map_at_k", title="MAP@k scores", color="search_type")
    plot_name = "mean_average_precision_at_k.html"
    mlflow_client.log_figure(run_id, fig, plot_name)


def plot_map_scores(df, run_id, mlflow_client):
    fig = px.bar(df, x="search_type", y="mean", title="MAP scores", color="search_type")
    plot_name = "mean_average_precision_scores.html"
    mlflow_client.log_figure(run_id, fig, plot_name)


def draw_search_chart(temp_df, run_id, mlflow_client):
    """
    Draws a comparison chart of search types across metric types.

    Args:
        temp_df (pandas.DataFrame): The dataframe containing the data to be plotted.
        run_id (int): The ID of the current run.
        mlflow_mlflow_client (mlflow.MlflowClient): The MLflow client to use for logging the chart.

    Returns:
        None
    """
    grouped = temp_df.groupby("search_type")
    summed_column = grouped.sum().reset_index()
    fig = sp.make_subplots(rows=len(summed_column.search_type), cols=1)
    for index, row_data in summed_column.iterrows():
        search_type = row_data[0]
        row_data = row_data[1:]
        df = row_data.reset_index(name="metric_value")
        df = df.rename(columns={"index": "metric_type"})
        fig.add_trace(
            go.Bar(
                x=df["metric_type"],
                y=df["metric_value"],
                name=search_type,
                offsetgroup=index,
            ),
            row=1,
            col=1,
        )

        fig.update_xaxes(title_text="Metric type", row=index + 1, col=1)
        fig.update_yaxes(title_text="score", row=index + 1, col=1)
    fig.update_layout(
        font=dict(size=15),
        title_text="Search type comparison across metric types",
        height=4000,
        width=800,
    )
    plot_name = "search_type_current_run.html"
    mlflow_client.log_figure(run_id, fig, plot_name)
