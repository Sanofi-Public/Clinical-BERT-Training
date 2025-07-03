import argparse
import os
import pprint
import tarfile
import yaml

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import TrainerCallback
from torch.nn.utils.rnn import pad_sequence
import tokenizers
import torch
from torch.utils.data import IterableDataset
from transformers import (
    EarlyStoppingCallback,
    IntervalStrategy,
    BertForMaskedLM,
    BertConfig,
    ModernBertForMaskedLM,
    ModernBertConfig,
    LongformerForMaskedLM,
    LongformerConfig,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
)

from data_processing.bert.loaders import CustomDataset, PretrainingDataCollator
from data_processing.bert.processing import MedbertDataPreprocessor
from data_processing.bert.callbacks import DelayedEarlyStoppingCallback, MetricsCallback, SaveMetricsCallback
from data_processing.bert.utils import get_vocab_from_file, resolve_sequence_length

from models.bert import compute_pretraining_metrics, preprocess_logits_for_metrics


def train_model(
    training_data_location,
    training_args,
    tokenizer,
    vocabulary,
    output_dir,
    code_filter=None,
    model=None,
    deepspeed=False,
):
    """
    Given a training data location on s3 or local disk, a validation data location on s3 or local disk,
    train a BERT model given the training_args yaml file found in config/training.yaml, for example.
    Also provide the tokenizer and vocabulary of the BERT model.

    training_data_location (string):
        the location of the training data. Either a string like s3://path/to/your/data, or a simple directory
        for where the data is on disk.
    validation_data_location (string):
        Same as above, but for the validation dataset.
    training_args:
        A dictionary of training arguments for the model. These arguments configure the training of the model.
        You'll probably have to look at the code to understand exactly what the arguments mean, or look at the comments
        in one of the config files like config/train.yaml.
    tokenizer (tokenizers.models.Model):
        The tokenizer model to handle tokenization
    vocabulary (dict of str: int):
        The mapping of tokens to the token_id. The input to a transformer model requires a series
        of token ids.
    output_dir (string):
        The directory to save the model in.
    skip_wandb (bool):
        Whether to skip the wandb reporting. This is needed in test cases, for example.
    code_filter:
        A comma separated list of strings to reduce what codes are used for training. If it is not None and set to "ICD10,AGE",
        then only the age and ICD10 tokens will be used during training.
    model:
        optional: a model to start trainining. If this is none, then create a new model to train
    """
    if model is None:  # the model was not supplied. Create it from the config.
        if training_args["model_flavour"] == "bert":
            config_class = BertConfig
            model_class = BertForMaskedLM
        elif training_args["model_flavour"] == "longformer":
            config_class = LongformerConfig
            model_class = LongformerForMaskedLM
        elif training_args["model_flavour"] == "modernbert":
            config_class = ModernBertConfig
            model_class = ModernBertForMaskedLM
        else:
            raise ValueError(
                f"Couldn't find the right kind of model for {training_args['model_flavour']}. Please extend the code above to fix."
            )

        # Initialize an untrained BERT model for Masked Language Modeling (MLM)
        config = config_class(
            vocab_size=len(vocabulary),  # Default vocab size for BERT
            hidden_size=training_args[
                "hidden_size"
            ],  # Default hidden size for BERT base model
            num_hidden_layers=training_args[
                "num_hidden_layers"
            ],  # Default number of layers for BERT base model
            num_attention_heads=training_args[
                "num_attention_heads"
            ],  # Default number of attention heads for BERT base model
            max_position_embeddings=training_args["max_position_embeddings"],
            hidden_dropout_prob=training_args["hidden_dropout_prob"],
            attention_probs_dropout_prob=training_args["attention_probs_dropout_prob"],
        )
        
        torch.cuda.empty_cache()
        model = model_class(config=config)
    
    if "override_maxlen" not in training_args:
        max_sequence_length = resolve_sequence_length(model)
    else:
        max_sequence_length = training_args["override_maxlen"]
    
    data_processor = MedbertDataPreprocessor(save=False)
    processed_diagnosis = data_processor.load(training_args['training_data_filepath'])
        
    train, val = train_test_split(
        processed_diagnosis,
        test_size=training_args['val_size'],
        random_state=42,
        shuffle=True,
    )
    print('Train: ', train.shape)
    print('Val: ', val.shape)

    train_dataset = CustomDataset(
        train,
        tokenizer,
        max_length=training_args['override_maxlen'],
        include_person_ids=True,
        shuffle=True,
    )
    
    val_dataset = CustomDataset(
        val,
        tokenizer,
        max_length=training_args['override_maxlen'],
        include_person_ids=True,
        shuffle=False,
    )

    args = TrainingArguments(
        # report_to=None,
        fp16=training_args["fp16"],
        bf16=training_args["bf16"],
        eval_strategy=IntervalStrategy.STEPS,
        eval_steps=training_args["eval_steps"],
        logging_steps=training_args["logging_steps"],
        learning_rate=training_args["learning_rate"],
        per_device_train_batch_size=training_args["batch_size"],
        per_device_eval_batch_size=training_args["eval_batch_size"],
        num_train_epochs=training_args["num_training_epochs"],
        weight_decay=training_args["weight_decay"],
        metric_for_best_model=training_args["early_stop_metric"],
        gradient_accumulation_steps=training_args["gradient_accumulation_steps"],
        output_dir=training_args['output'],
        logging_strategy="steps",
        logging_dir=output_dir,
        load_best_model_at_end=True,
        eval_accumulation_steps=1,
        save_steps=training_args["save_steps"],
        save_strategy="steps",
        save_total_limit=training_args[
            "save_total_limit"
        ],  # prevent to many models from being saved
        dataloader_num_workers=training_args["num_dataloader_workers"],
    )

    data_collator = PretrainingDataCollator(
        tokenizer,
        vocabulary,
        prediction_fraction=training_args["prediction_fraction"],
        masking_fraction=training_args["masking_fraction"],
        random_replacement_fraction=training_args["random_replacement_fraction"],
        override_maxlen=training_args['override_maxlen'],
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=training_args[
            "early_stop_patience"
        ]  # Number of evaluations with no improvement after which training will be stopped
    )
    delayed_early_stopping_callback = DelayedEarlyStoppingCallback(
        early_stopping_callback, delay_epochs=training_args["early_stopping_delay"]
    )

    metrics_callback = MetricsCallback(output_dir)
    logging_callback = SaveMetricsCallback(output_dir)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_pretraining_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[delayed_early_stopping_callback, metrics_callback, logging_callback],
    )
    train_result = trainer.train()

    # Save training metrics
    trainer.save_metrics("all", train_result.metrics)

    return model, args, trainer  # return everything needed to save the model.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for BERT inference script.")
    parser.add_argument('-c', '--config-path', type=str, default='configs/pretrain.yaml', help='Path to the BERT configuration file')
    
    args = parser.parse_args()
    
    with open(args.config_path, 'r') as file:
        pretrain_config = yaml.safe_load(file)
        pprint.pp(pretrain_config)
        
    local_save_directory = pretrain_config['output']

    head, tail = os.path.split(local_save_directory)
    local_training_logs_directory = os.path.join(head, tail + '_training_logs')
    print(f"Saving training logs in {local_training_logs_directory}")
    
    vocab_filepath = pretrain_config['vocab_filepath']
    tokenizer = tokenizers.models.WordPiece.from_file(vocab_filepath)
    vocabulary = get_vocab_from_file(vocab_filepath)
    
    model, pretrain_config, trainer = train_model(
        pretrain_config['training_data_filepath'],
        pretrain_config,
        tokenizer,
        vocabulary,
        local_training_logs_directory,
        code_filter=None, # args["code_filter"]
    )

    model.save_pretrained(local_save_directory)
    tokenizer.save(local_save_directory)

    # zip-up the folder
    head, tail = os.path.split(local_save_directory)
    tar_filename = os.path.join(head, tail + ".tar.gz")
  
    try:
        with tarfile.open(tar_filename, "w") as tar:
            tar.add(
                local_save_directory, arcname=os.path.basename(local_save_directory)
            )
    except Exception as e