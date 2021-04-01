import pandas as pd
from pandas import DataFrame
import numpy as np


class FeatureEngineering:

    def __init__(self, df: DataFrame):

        self.df = df

    def calculate_age(self):

        self.df['age'] = np.where(self.df.construction_year != 0,
                                  self.df.date_recorded.dt.year - self.df.construction_year,
                                  np.nan)

        return self.df
