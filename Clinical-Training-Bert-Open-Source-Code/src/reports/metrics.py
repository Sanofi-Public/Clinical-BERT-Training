from typing import Optional, Dict, Union, Tuple, List

from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score

from sklearn.metrics import log_loss, brier_score_loss


class ModelMetricsGenerator:
    """A class for calculating various performance metrics for a binary classification model."""

    def __init__(
        self,
        y_true: NDArray,
        pred_proba: NDArray,
        threshold: float = 0.5,
        prevalence: float = 1 / 200,
        tag: Optional[str] = None,
    ):
        """
        Args:
            y_true (NDarray): Ground truth binary labels for the data.
            pred_proba (NDarray): Predicted probabilities.
            threshold (float): The threshold value for converting predicted probabilities into binary labels.
            prevalence (float): The prevalence rate of the condition in the population, used in the calculation of certain metrics.
            tag (Optional[str]): An optional string to tag metric names for models differentiation.
        """
        self.y_true = y_true
        self.pred_proba = pred_proba
        self.y_pred = self.pred_proba > threshold

        self.calculate_confusion_matrix()
        self.tag = tag
        self.prevalence = prevalence

    def calculate_confusion_matrix(self) -> None:
        """
        Computes the confusion matrix and calculates the positive predictive value (PPV).
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        tn, fp, fn, tp = cm.ravel()

        self.confusion_matrix = cm
        self.tn = tn
        self.fp = fp
        self.fn = fn
        self.tp = tp

        self.ppv = self.__safe_div(tp, tp + fp)

    def get_metrics(self) -> Dict[str, float]:
        """
        Computes various performance metrics for the model.

        Returns:
            Dict[str, float]: A dictionary with calculated metrics. The dictionary keys will be suffixed with the provided `tag` if specified.
        """
        tp_fn_sum = self.tp + self.fn
        fp_tn_sum = self.fp + self.tn
        
        tpr = self.__safe_div(self.tp, tp_fn_sum)
        fpr = self.__safe_div(self.fp, fp_tn_sum)
        
        p, r, _ = precision_recall_curve(self.y_true, self.y_pred)
        pr_auc = auc(r, p)

        sensitivity = self.__safe_div(self.tp, tp_fn_sum) 
        specificity = self.__safe_div(self.tn, fp_tn_sum)
        
        gmean = (sensitivity * specificity) ** 0.5

        p_t1d_given_positive = (tpr * self.prevalence) / (
            tpr * self.prevalence + fpr * (1 - self.prevalence) + 1e-8
        )
        bayes_factor = sensitivity / (1 - specificity + 1e-8)

        metrics = {
            'confusion_matrix': self.confusion_matrix,
            'precision': precision_score(self.y_true, self.y_pred),
            'recall': recall_score(self.y_true, self.y_pred),
            'f_score': f1_score(self.y_true, self.y_pred),
            'roc_auc': roc_auc_score(self.y_true, self.y_pred),
            'pr_auc': pr_auc,
            'balanced_accuracy': balanced_accuracy_score(self.y_true, self.y_pred),
            'sensitivity': sensitivity,
            'specificity': specificity,
            'gmean': gmean,
            'p_t1d_given_positive': p_t1d_given_positive,
            'bayes_factor': bayes_factor,
            'log_loss': log_loss(self.y_true, self.y_pred),
            'brier_score_loss': brier_score_loss(self.y_true, self.y_pred),
            'tpr': tpr,
            'fpr': fpr,
        }

        if self.tag:
            metrics = {
                metric_name + f"_{self.tag}": value
                for metric_name, value in metrics.items()
            }

        rounded_metrics = dict()
        for metric_name, value in metrics.items():
            new_value = round(value, 2) if isinstance(value, float) else value
            rounded_metrics[metric_name] = new_value

        return rounded_metrics

    
    @staticmethod
    def __safe_div(numerator, denominator):
        """Safely divides numerator by denominator, returning 0 if denominator is zero."""
        return numerator / denominator if denominator else 0

        
class MultipleModelsReportGenerator:
    """A class for calculating various performance metrics for a multiple binary classification models."""

    def __init__(self, report_data, report_data_random, report_data_const):
        self.report_data = report_data
        self.report_data_random = report_data_random
        self.report_data_const = report_data_const

        self.metrics = self.report_data.metrics_generator.get_metrics()
        self.metrics_random = self.report_data_random.metrics_generator.get_metrics()
        self.metrics_const = self.report_data_const.metrics_generator.get_metrics()

    def get_metrics(self) -> Dict:
        """
        Computes comparative metrics for all models.

        Returns:
            Dict: A dictionary with metrics.
        """
        bss = self.get_brier_skill_scores()

        metrics = {
            "brier_skill_scores": bss
        }

        return metrics


    @staticmethod
    def __get_points(
        x: NDArray,
        y: NDArray,
    ) -> Tuple[Tuple[float], Tuple[float]]:
        """
        Retrieves two endpoints from a given set of x and y coordinates.

        Args:
            x: An array of x-coordinates.
            y: An array of y-coordinates.

        Returns:
            Tuple[Tuple[float], Tuple[float]]: A tuple containing two endpoints ((x1, y1), (x2, y2)).
        """
        point = x[0], y[0]
        other_point = x[-1], y[-1]

        return point, other_point


    def get_brier_skill_scores(self):
        """
        Calculates Brier Skill Scores for the primary model and compares them against reference models.

        Returns:
            Dict: A dictionary containing Brier Skill Scores for the primary model, random model, and constant model.
        """
        model = self.metrics["brier_score_loss"]
        ref = self.metrics_random["brier_score_loss_random"]
        const = self.metrics_const["brier_score_loss_const"]

        scores = {
            "brier_skill_score": self._calculate_bss(model, ref),
            "brier_skill_score_random": "n/a",
            "brier_skill_score_const": self._calculate_bss(const, ref),
        }

        return scores

    @staticmethod
    def _calculate_bss(current: float, reference: float) -> float:
        """
        Calculates the Brier Skill Score for the current model against a reference model.

        Args:
            current (float): The Brier score loss of the current model.
            reference (float): The Brier score loss of the reference model.

        Returns:
            float: Brier skill score value.
        """
        return round(1 - current / reference, 4)

