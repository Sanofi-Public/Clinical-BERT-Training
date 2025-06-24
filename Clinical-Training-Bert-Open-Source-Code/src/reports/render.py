import os
from itertools import chain
from typing import Dict, List, Tuple, Optional

from sklearn.metrics import precision_recall_curve, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from jinja2 import Environment, FileSystemLoader

from reports.containers import MultipleModelsReportData


class ReportRenderer:
    """A class for generating and rendering model performance reports, including plots."""

    def __init__(
        self,
        models_report: MultipleModelsReportData,
        metadata: Dict,
        configs: Dict,
        tags: List[str],
    ):
        """
        Args:
            models_report (MultipleModelsReportData): An instance containing comparative metrics of multiple models.
            metadata (Dict): A dictionary containing report metadata.
            configs (Dict): A dictionary containing configuration details or parameters used by the model.
            tags (List[str]): A list of tags identifying the models.
        """
        self.models_report = models_report

        self.metadata = metadata
        self.configs = configs
        self.tags = tags

        self.cm_filename = "confusion_matrix"
        self.report_dir = os.path.join(
            "experiments", metadata["report_datetime"].replace(":", "-")
        )
        self.report_artifacts_dir = os.path.join(self.report_dir, "artifacts")

    @property
    def images_filepath(self) -> Dict[str, str]:
        """
        A dictionary with file paths for various report images.

        Returns:
            Dict[str, str]: Report images filepaths.
        """
        return {
            "confusion_matrix_image": "artifacts/confusion_matrix.png",
            "confusion_matrix_image_ref": "artifacts/confusion_matrix_random.png",
            "confusion_matrix_image_const": "artifacts/confusion_matrix_const.png",
            "pr_curve_image": "artifacts/pr_curve.png",
            "pr_curve_image_ref": "artifacts/pr_curve_random.png",
            "pr_curve_image_const": "artifacts/pr_curve_const.png"
        }

    def __create_report_dir(self) -> None:
        """
        Creates directories for saving the report and its associated artifacts.
        """
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.report_artifacts_dir, exist_ok=True)

    def render(self, save: bool = True) -> None:
        """
        Renders the HTML report using a Jinja2 template and optionally saves it to disk.

        Args:
            save (bool): If True, saves the generated HTML report and plots to disk (default is `True`).
        """
        env = Environment(loader=FileSystemLoader("."))
        template = env.get_template("reports/report_template.html")

        multimodel_metrics = self.models_report.models_comparison_metrics
        bss = multimodel_metrics["brier_skill_scores"]

        notes = "n/a"

        html_content = template.render(
            notes=notes,
            model_params=self.configs,
            **self.metadata,
            **self.images_filepath,
            **self.models_report.report_data.metrics,
            **self.models_report.report_data_random.metrics,
            **self.models_report.report_data_const.metrics,
            **bss,
        )

        if save:
            self.save(html_content)

            for model_metrics, tag in zip(self.models_report.metrics, self.tags):
                self.plot_confusion_matrix(model_metrics, tag=tag)
                self.plot_pr_curve(tag=tag)

        else:
            print(html_content)

    def save(self, html_content: str) -> None:
        """
        Saves the HTML content to a file.

        Args:
            html_content (str): The HTML content to be saved to a file.
        """
        self.__create_report_dir()

        filepath = os.path.join(self.report_dir, "model_report.html")
        with open(filepath, "w") as f:
            f.write(html_content)

        print(f"Report generated: {filepath}")

    def plot_confusion_matrix(
        self,
        model_metrics: Dict[str, float],
        tag: Optional[str] = None,
    ) -> None:
        """
        Plots and saves the confusion matrix for a given model.

        Args:
            model_metrics (Dict[str, float]): A dictionary containing the metrics of the individual model.
            tag (Optional[str]): A tag identifying the model (default is `None`).
        """
        cm_name = f"{self.cm_filename}_{tag}" if tag else self.cm_filename
        cm = model_metrics[cm_name]

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["no-T1D", "T1D"]
        )
        disp.plot(cmap=plt.cm.Blues)
        title = f"{tag.title()}" if tag else None
        plt.title(title)
        plt.savefig(os.path.join(self.report_artifacts_dir, f"{cm_name}.png"))
        plt.close()

    def plot_pr_curve(self, tag: Optional[str] = None) -> None:
        """
        Plots and saves the precision-recall curve for a given model.

        Args:
            tag (A tag identifying the model (default is None)): A tag identifying the model (default is `None`).
        """
        predictions = {
            "random": self.models_report.report_data_random.predictions,
            "const": self.models_report.report_data_const.predictions,
        }

        y_pred = predictions.get(tag, self.models_report.report_data.predictions)

        precision, recall, _ = precision_recall_curve(
            self.models_report.report_data.ground_truth, y_pred
        )

        filename = f"pr_curve_{tag}.png" if tag else "pr_curve.png"
        title = f"{tag.title()}" if tag else None

        t1d_ratio = self.models_report.report_data.t1d_ratio
        plt.plot([0, 1], [t1d_ratio, t1d_ratio], linestyle="--", label="No Skill")
        plt.plot(recall, precision, marker=".", label="Model")
        plt.title(title)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()

        plt.savefig(os.path.join(self.report_artifacts_dir, filename))
        plt.close()

