from idata_peprocessor import IDataPreprocessor

import pandas as pd
import numpy as np

import sklearn.preprocessing as preprocessing

class CPDataPreprocessor(IDataPreprocessor):
    @staticmethod
    def preprocess(data) -> pd.DataFrame:
        """
        
        """