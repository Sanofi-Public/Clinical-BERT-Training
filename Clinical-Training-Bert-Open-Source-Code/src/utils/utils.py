import json
from typing import Dict


def read_config(filepath: str) -> Dict:
    """
    Reads json config.

    Args:
        filepath (str): Config filepath.

    Returns:
        Dict: Config dict.
    """
    with open(filepath, "r") as file:
        config = json.load(file)

    return config
