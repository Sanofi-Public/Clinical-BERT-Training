import json
import pickle
import os
from datetime import datetime
from typing import Dict, Tuple, Optional

import torch
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import (
    explained_variance_score, 
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    d2_absolute_error_score,
)

from torch.utils.data import DataLoader
from tqdm import tqdm

from models.bert import BertProblems
from models.dummy import RandomClassifier, ConstantClassifier

from reports.render import ReportRenderer
from reports.containers import ModelReportData, MultipleModelsReportData


class BertRegressionReport:
    def __init__(self, model, dataloader, problem_type, device=None, batch_size=24):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.problem_type = problem_type
        self.batch_size = batch_size
    
    def generate(self, save=True):
        report_datetime = str(datetime.now().replace(microsecond=0))
        report_dir = os.path.join('experiments', report_datetime.replace(':', '-'))
        
        os.makedirs(report_dir, exist_ok=True)
        
        report = get_report(
            self.model,
            self.dataloader,
            device=self.device,
            problem_type=self.problem_type,
            batch_size=self.batch_size,
            save_dir=os.path.join(report_dir),
        )
        
        report_df = pd.DataFrame(report)
        
        if save:
            report_df.to_csv(os.path.join(report_dir, 'report.csv'), index=False)
            print(f"Report generated: {report_dir}")
        
        return report_df


def get_report(
    model,
    dataloader,
    problem_type=None,
    device=None,
    batch_size=16,
    save_dir=None,
):
    all_predictions = []
    all_labels = []
    all_person_ids = []
    sm = torch.nn.Softmax(dim=1)
    
    for batch in tqdm(
        dataloader,
        desc='Processing Batches',
        total=len(dataloader) // batch_size,
    ):
        with torch.no_grad():
            batch, person_id = {
                key: batch[key].to(device) for key in batch if key != 'person_id'
            }, batch['person_id']
            result = model(**batch)
            
            if problem_type == BertProblems.REGRESSION:
                predictions = result.logits.cpu().numpy()
            else:
                predictions = sm(result.logits).cpu().numpy()
                
            all_predictions.append(predictions)
            all_labels.append(batch['labels'].cpu().numpy())
            all_person_ids.append(np.array(person_id))

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    all_person_ids = np.concatenate(all_person_ids)
    
    with open(os.path.join(save_dir, 'predictions.pkl'), 'wb') as f:
        pickle.dump(
            {
                'predictions': all_predictions,
                'labels': all_labels,
                'person_ids': all_person_ids,
            },
            f,
        )

    if problem_type == BertProblems.REGRESSION:      
        report = {}
        report['model'] = get_regression_report(all_labels, all_predictions)
        
        for strategy in ['mean', 'median']:
            dummy = DummyRegressor(strategy=strategy)
            dummy.fit(np.zeros((len(all_labels), 1)), all_labels)  # Dummy input
            dummy_preds = dummy.predict(np.zeros((len(all_labels), 1)))

            report[f'dummy_{strategy}'] = get_regression_report(all_labels, dummy_preds)
    else:
        pred_label = np.argmax(all_predictions, axis=1)    
        report = classification_report(all_labels, pred_label, output_dict=True)
    
    report_df = pd.DataFrame(report)
    report_df.reset_index(inplace=True)
    report_df.rename(columns={'index': 'metrics'}, inplace=True)
    
    return report_df.round(6)

        
def get_regression_report(y_true, y_pred):
    explained_variance = explained_variance_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    median_ae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    d2_ae = d2_absolute_error_score(y_true, y_pred)
    
    report = {
        'explained_variance': explained_variance,
        'mean_ae': mae,
        'mse': mse,
        'rmse': np.sqrt(mse),
        'median_ae': median_ae,
        'r2': r2,
        'd2_ae': d2_ae,
    }

    return report


class EnsembleEvaluationReport:
    """
    A class to generate evaluation reports comparing multiple models, including
    a custom model, a random stratified classifier, and a constant classifier.
    """

    def __init__(self, scaler: StandardScaler, configs: Dict, model_preds: NDArray):
        """
        Args:
            scaler (StandardScaler): The scaler used for preprocessing the data.
            configs (Dict): Configuration parameters for the model being evaluated.
            model_preds (NDArray): Predictions made by the model to be evaluated.
        """
        self.scaler = scaler
        self.configs = configs
        self.model_preds = model_preds

        self.model_tags = [None, "random", "const"]
        self.report_dir = None

    def generate(
        self,
        train_data: Tuple[NDArray, NDArray],
        test_data: Tuple[NDArray, NDArray],
        save: bool = True,
    ) -> None:
        """
        Generates the evaluation report by comparing the custom model against
        random and constant classifiers.

        Args:
            train_data (Tuple[NDArray, NDArray]): The training dataset, consisting of features (x_train) and labels (y_train).
            test_data (Tuple[NDArray, NDArray]): The test dataset, consisting of features (x_test) and labels (y_test).
            save (bool): If True, saves the generated report to disk (default is `True`).
        """
        x_train, y_train = train_data
        x_test, y_test = test_data

        random_clf = RandomClassifier(self.scaler)
        constant_clf = ConstantClassifier(self.scaler)

        random_clf.fit(x_train, y_train)
        constant_clf.fit(x_train, y_train)

        random_preds = random_clf.predict(x_test)
        const_preds = constant_clf.predict(x_test)
        preds = {"random": random_preds, "const": const_preds}

        reports_data = []
        t1d_ratio = sum(y_test) / y_test.shape[0]
        for tag in self.model_tags:
            y_pred = preds.get(tag, self.model_preds)
            report_data = ModelReportData(y_test, y_pred, tag=tag, t1d_ratio=t1d_ratio)
            reports_data.append(report_data)

        models_report = MultipleModelsReportData(*reports_data, self.model_tags)

        train_dataset_size = x_train.shape[0]
        test_dataset_size = x_test.shape[0]
        model_params = json.dumps(self.configs, indent=4)

        metadata = {
            "report_datetime": str(datetime.now().replace(microsecond=0)),
            "train_dataset_size": train_dataset_size,
            "test_dataset_size": test_dataset_size,
            "train_t1d_ratio": round(y_train.sum() / train_dataset_size, 6),
            "test_t1d_ratio": round(y_test.sum() / test_dataset_size, 6),
        }

        report_generator = ReportRenderer(
            models_report,
            metadata,
            model_params,
            self.model_tags,
        )
        self.report_dir = report_generator.report_dir

        report_generator.render(save=save)


class BertInferenceEvaluationReport:
    """
    A class to generate evaluation reports comparing multiple models, including
    a BERT model, a random stratified classifier, and a constant classifier.
    """
    def __init__(self, model_preds: NDArray, model_name: Optional[str] = None):
        """
        Args:
            model_preds (NDArray): Predictions made by the model to be evaluated.
        """
        self.model_preds = np.array(model_preds)
        self.model_name = model_name
        
        self.model_tags = [None, 'random', 'const']
        self.report_dir = None
    
    @staticmethod
    def get_dataloader_labels(dataloader):
        labels = []
        for batch in dataloader:
            batch_labels = batch['labels']
            labels.extend(torch.squeeze(batch_labels, axis=1))
        
        return np.array(labels)
            
    def generate(
        self,
        test_dataloader,
        predictions_df,
        save: bool = True,
    ) -> None:
        """
        Generates the evaluation report by comparing the custom model against random and constant classifiers.
        
        Args:
            test_dataloader (DataLoader):
            predictions_df (pd.DataFrame):
            save (bool): If True, saves the generated report to disk (default is `True`).
        """
        if isinstance(test_dataloader, DataLoader):
            y_test = self.get_dataloader_labels(test_dataloader)
        elif isinstance(test_dataloader, np.ndarray):
            y_test = test_dataloader
        else:
            raise ValueError(f'Incorrect data loader type: {type(test_dataloader)}')
        
        random_clf = RandomClassifier()
        constant_clf = ConstantClassifier()

        random_clf.fit(None, y_test)
        constant_clf.fit(None, y_test)

        random_preds = random_clf.predict(np.zeros_like(y_test))
        const_preds = constant_clf.predict(np.zeros_like(y_test))

        preds = {'random': random_preds, 'const': const_preds}

        reports_data = []
        t1d_ratio = sum(y_test) / len(y_test)
        for tag in self.model_tags:
            y_pred = preds.get(tag, self.model_preds)
            report_data = ModelReportData(y_test, y_pred, tag=tag, t1d_ratio=t1d_ratio)
            reports_data.append(report_data)

        models_report = MultipleModelsReportData(*reports_data, self.model_tags)

        train_dataset_size = 0
        test_dataset_size = y_test.shape[0]

        metadata = {
            'report_datetime': str(datetime.now().replace(microsecond=0)),
            'train_dataset_size': train_dataset_size,
            'test_dataset_size': test_dataset_size,
            'train_t1d_ratio': 'n/a',
            'test_t1d_ratio': round(y_test.sum() / test_dataset_size, 6),
        }
        
        report_generator = ReportRenderer(
            models_report,
            metadata,
            {'model_name': self.model_name},
            self.model_tags,
        )
        self.report_dir = report_generator.report_dir
        
        report_generator.render(save=save)
        if save and predictions_df is not None:
            predictions_df.to_csv(
                os.path.join(self.report_dir, 'predictions.csv'),
                index=False,
            )
