import argparse
import json
import os
import pprint
import tarfile
import itertools
import logging
import sys
import shutil

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import transformers
import tokenizers

from tqdm import tqdm

from torch.utils.data import DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import torch.nn.functional as F
from torch.nn import Softmax

import torch
import tokenizers
from transformers import (
    EarlyStoppingCallback,
    TrainerCallback,
    IntervalStrategy,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from models.bert import get_metrics_function, BertProblems
from data_processing.bert.loaders import CustomDataset, LabelledDataset, LabelledDataCollator
from data_processing.bert.utils import get_vocab_from_file, get_codes_mapping
from data_processing.bert.processing import MedbertDataPreprocessor

from reports import BertInferenceEvaluationReport
from reports.report import get_report

from utils import read_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for BERT inference script.")
    
    parser.add_argument('-c', '--config-path', type=str, default='configs/bert.json', help='Path to the BERT configuration file')
    args = parser.parse_args()

    bert_config = read_config(args.config_path)
    pprint.pp(bert_config)
    
    target_column = bert_config['target_column']
    problem_type = bert_config['problem_type']
    pretrained_model_dir = bert_config['pretrained_model_dir']
    finetuned_model_name = bert_config['finetuned_model_name']
    
    codes_mapping = get_codes_mapping(os.path.join(pretrained_model_dir, 'vocab.txt'))
        
    data_processor = MedbertDataPreprocessor(save=False)
    processed_diagnosis = data_processor.load(bert_config['data_filepath'])
    
    tokenizer = tokenizers.models.WordPiece.from_file(os.path.join(pretrained_model_dir, 'vocab.txt'))
    vocabulary = get_vocab_from_file(os.path.join(pretrained_model_dir, 'vocab.txt'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = transformers.BertForSequenceClassification.from_pretrained(
        pretrained_model_dir,
        problem_type=problem_type,
        num_labels=bert_config['num_labels'],
        ignore_mismatched_sizes=True,
    ).to(device)

    stratification_col = None if problem_type == 'regression' else processed_diagnosis[target_column]
    
    train, val = train_test_split(
        processed_diagnosis,
        test_size=bert_config['test_size'],
        random_state=bert_config['random_state'],
        shuffle=True,
        stratify=stratification_col,
    )
    print('Train: ', train.shape)
    print('Val: ', val.shape)
    
    train_dataset = LabelledDataset(
        target_column,
        train,
        tokenizer,
        max_length=bert_config['max_length'],
        include_person_ids=True,
    )
    
    val_dataset = LabelledDataset(
        target_column,
        val,
        tokenizer,
        max_length=bert_config['max_length'],
        include_person_ids=True,
    )

    config = bert_config['training_args']
    training_args = TrainingArguments(
        eval_strategy=IntervalStrategy.STEPS,
        save_strategy=IntervalStrategy.STEPS,
        **config,
    )
    
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=bert_config['patience'],
    )

    # Initialize the Trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=LabelledDataCollator(), 
        compute_metrics=get_metrics_function(problem_type),
        callbacks=[early_stopping_callback],
    )
    
    report_save_dir = os.path.join('outputs/', finetuned_model_name)

    if not os.path.exists(report_save_dir):
        os.makedirs(report_save_dir)
        
    train_results = trainer.train()    
    history = pd.DataFrame(trainer.state.log_history)
    history.to_csv(os.path.join(report_save_dir, 'history.csv'), index=False)

    # saving train dataset metrics; `run_bert_inference.py` is running evaluation on test data 
    test_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['per_device_train_batch_size'],
        collate_fn=LabelledDataCollator(),
    )
    
    report = get_report(
        model,
        test_dataloader,
        device=device,
        problem_type=problem_type,
        batch_size=config['per_device_train_batch_size'],
        save_dir=report_save_dir,
    )

    print(report)
    
    if problem_type != BertProblems.REGRESSION:
        report.rename(columns={'index': 'class'}, inplace=True)
    
    report.to_csv(os.path.join(report_save_dir, 'train_report.csv'), index=False)
    
    # save the model and archive it
    model.save_pretrained(report_save_dir)
    # os.system(f"cp {pretrained_model_dir}/vocab.txt {report_save_dir}/vocab.txt")
    shutil.copy(
        os.path.join(pretrained_model_dir, "vocab.txt"),
        os.path.join(report_save_dir, "vocab.txt")
    )
