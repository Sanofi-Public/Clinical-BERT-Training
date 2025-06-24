from typing import Dict

import pandas as pd


def get_codes_mapping(vocab_file: str) -> Dict[str, str]:
    """
    Creates ICD codes mapping from model vocab file.
    
    Args:
        vocab_file (str): Model vocab filepath.
        
    Return:
        Dict[str, str]: ICD codes mapping.
    """
    mapping = {}

    with open(vocab_file, "r") as f:
        for el in f.readlines():
            token = el.replace("\n", "")
            try:
                icd, code = token.split(':')
            except:
                print(token)
                
            mapping[code] = icd
    
    return mapping


def get_vocab_from_file(vocab_file: str) -> Dict[str, int]:
    """
    Loads vocab file and creates vocab mapping.
    
    Args:
        vocab_file (str): Model vocab filepath.
    
    Returns:
        Dict: `token` -> `index` mapping.
    """
    with open(vocab_file, "r") as f:
        vocab = {el.replace("\n", ""): i for i, el in enumerate(f.readlines())}
    
    return vocab


def resolve_sequence_length(model) -> int:
    """
    Determines the maximum sequence length for a given model type.

    Args:
        model: A model instance (e.g., BertModel, LongformerModel).

    Returns:
        int: The maximum sequence length supported by the model.

    """
    if isinstance(model, (BertModel, BertForMaskedLM)):
        max_sequence_length = 512
    elif isinstance(model, (LongformerModel, LongformerForMaskedLM)):
        max_sequence_length = 4092
    else:
        raise ValueError(
            f"Unknown model type {model}. Please extend the code in {__file__} by specifying the max sequence length for the model."
        )
    
    return max_sequence_length
