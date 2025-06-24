import os
from collections import defaultdict, OrderedDict
from dateutil.relativedelta import relativedelta
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class MedbertDataPreprocessor:
    def __init__(self, *args, **kwargs):
        pass
        
    def process(self):
        """
        Performs data processing and convertation to MedBERT format.
            
        Returns:
            pd.DataFrame: Processed dataframe. 
        """
        pass
    
    def load(self, filepath: str, engine='pyarrow') -> pd.DataFrame:
        """
        Loads DataFrame from parquet file.

        Args:
            filepath (str): Parquet dataset file path.
            engine (str): Parquet engine.
            
        Returns:
            (pd.DataFrame): Loaded dataframe.
        """
        print('Loading preprocessed data...')
        # todo: fix parquet file loading
        import polars as pl
        
        df_polars = pl.read_parquet(filepath)
        processed_dataset = df_polars.to_pandas()
            
        del df_polars
        
        # processed_dataset = pd.read_parquet(filepath, engine=engine)        
        
        return processed_dataset
