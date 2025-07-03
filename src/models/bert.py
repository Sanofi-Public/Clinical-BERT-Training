from enum import Enum

import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import Trainer


class BertProblems(str, Enum):
    CLASSIFICATION = 'single_label_classification'
    REGRESSION = 'regression'
    

def compute_classification_metrics(predictions=None):
    preds = np.argmax(predictions.predictions, axis=1)
    scores = softmax(predictions.predictions, axis=1)
    labels = predictions.label_ids[:, 0]
    all_possible_labels = np.unique(labels)
    roc_scores = {}
    for one_label_vs_rest in all_possible_labels:
        one_vs_rest_labels = 1 * (labels == one_label_vs_rest)
        one_vs_rest_scores = scores[:, one_label_vs_rest]
        roc_scores[f"eval_roc_auc_{one_label_vs_rest}_vs_rest"] = roc_auc_score(
            one_vs_rest_labels, one_vs_rest_scores
        )

    report = classification_report(labels, preds, output_dict=True)
    flatten_dict = {}  # a dictionary to hold all metrics
    for key in report:
        if isinstance(report[key], dict):
            for key2 in report[key]:
                flatten_dict[f"eval_{key}_{key2}"] = report[key][key2]
        else:
            flatten_dict[f"eval_{key}"] = report[key]

    flatten_dict.update(roc_scores)
    average_roc = sum([roc_scores[el] for el in roc_scores]) / len(roc_scores)
    flatten_dict["eval_roc_auc_average"] = average_roc

    return flatten_dict


def compute_regression_metrics(predictions=None):
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids[:, 0]
    
    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
    }


def get_metrics_function(task_name=None):
    functions = {
        'single_label_classification': compute_classification_metrics,
        'regression': compute_regression_metrics,
    }

    try:
        function = functions[task_name]
        return function
    except KeyError as e:
        print(f'Incorrect task {task_name}. Please, select one of the following: {list(functions.keys())}')


def preprocess_logits_for_metrics(logits, labels):
    """
    Given a np.array of logits of shap batch_size x sequence_length x vocab_size
    and labels of size batch_size x sequence_length,
    preprocess the logits by selecting those that are predictions
    for masked language modelling (labels != -100).
    Afterwards, select the argmax along the vocabulary axis to select the
    predicted token. return the predictions and labels
    """
    predictions = torch.argmax(logits, -1)
    return predictions  # , labels


def compute_pretraining_metrics(eval_pred):
    """
    Given a tuple of arbitrary length, assume that the first element
    are the clean predictions and labels from the above function.
    Calculate the accuracy of how many tokens are correctly predicted.
    """
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    mask = labels > -50
    predictions, labels = predictions[mask], labels[mask]
    
    return {"accuracy": np.sum(1 * (predictions == labels)) / len(predictions)}


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_fct = kwargs.pop("loss_fct")
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)  # added loss
        logits = outputs.get("logits")
        loss = self.loss_fct(logits, labels[:, 0])
        return (loss, outputs) if return_outputs else loss
