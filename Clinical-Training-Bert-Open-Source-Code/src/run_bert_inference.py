import argparse
import json
import os
import pprint
import tarfile
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import transformers
import tokenizers

from data_processing.bert.loaders import LabelledDataset, LabelledDataCollator
from data_processing.bert.utils import get_vocab_from_file, get_codes_mapping
from data_processing.bert.processing import MedbertDataPreprocessor

from models.bert import BertProblems
from inference import BertInferencePipeline
from reports import BertInferenceEvaluationReport, BertRegressionReport
from reports.report import get_report
from utils import read_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for BERT inference script.")
    
    parser.add_argument('-c', '--config-path', type=str, default='configs/bert.json', help='Path to the BERT configuration file')
    
    args = parser.parse_args()
    
    # Load BERT config
    bert_config = read_config(args.config_path)
    pprint.pp(bert_config)
    
    target_column = bert_config['target_column']
    problem_type = bert_config['problem_type']
    pretrained_model_dir = bert_config['pretrained_model_dir']
    finetuned_model_dir = bert_config['finetuned_model_name']
    
    # Preprocess data
    finetuned_dir_root = os.path.join('outputs', finetuned_model_dir)
    codes_mapping = get_codes_mapping(os.path.join(finetuned_dir_root, 'vocab.txt'))
    
    data_processor = MedbertDataPreprocessor(save=False)
    processed_diagnosis = data_processor.load(bert_config['test_data_filepath'])
    
    # Creating tokenizer and data loader
    tokenizer = tokenizers.models.WordPiece.from_file(os.path.join(finetuned_dir_root, 'vocab.txt'))
    vocabulary = get_vocab_from_file(os.path.join(finetuned_dir_root, 'vocab.txt'))
    
    test_dataset = LabelledDataset(
        bert_config['target_column'],
        processed_diagnosis,
        tokenizer,
        max_length=bert_config['max_length'],
        include_person_ids=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=24,
        collate_fn=LabelledDataCollator(),
        pin_memory=True,
    )
    
    # loading model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = transformers.BertForSequenceClassification.from_pretrained(finetuned_dir_root)
    model.to(device)
    
    # Running inference
    pipeline = BertInferencePipeline(model, config=bert_config, device=device)
    result = pipeline.run(test_dataloader)
    
    predictions_df = pd.DataFrame(result)
    
    if problem_type == BertProblems.REGRESSION: 
        report = BertRegressionReport(
            model,
            test_dataloader,
            problem_type=problem_type,
            device=device,
        )
        report.generate(save=True)
    else:
        report = BertInferenceEvaluationReport(predictions_df['predictions'], model_name=finetuned_model_dir)
        report.generate(test_dataloader, predictions_df, save=True)
