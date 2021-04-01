"""
This file contains custom sklearn estimators and transformers that work with pandas dataframes and are pipeline compatible.
Normal sklearn functionality works with numpy. These classes extend model data preparation to pandas objects
"""
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict
from pandas import DataFrame
from numpy import ndarray
import numpy as np
import pandas as pd


class DropFeatures(BaseEstimator, TransformerMixin):

    """
    Drops columns not required for modelling.
    """

    def __init__(self, variables_to_drop: List[str] = None):

        self.variables = variables_to_drop

    def fit(self, x, y=None):
        return self

    def transform(self, x: DataFrame) -> DataFrame:

        x = x.drop(self.variables, axis=1)

        return x


class NumericalMedianImputer(BaseEstimator, TransformerMixin):

    """
    Imputes median values of specified columns if records contain null values.

    The fitting method stores the median for each variable.

    The transform method performs imputation
    """

    def __init__(self, variables: List[str] = None):

        self.imputer_dict_ = {}

        if not isinstance(variables, list):

            self.variables = [variables]

        else:

            self.variables = variables

    def fit(self, x: DataFrame, y=None) -> Dict:

        for col in self.variables:

            self.imputer_dict_[col] = x[col].median()

        return self

    def transform(self, x: DataFrame) -> DataFrame:

        for col in self.variables:

            x[col] = x[col].fillna(self.imputer_dict_[col])

        return x


class FillNumericNullsWithZero(BaseEstimator, TransformerMixin):

    """
    Fills null values with 0 for specified columns
    """

    def __init__(self, variables: List[str] = None):

        self.variables = variables

    def fit(self, x, y=None):

        return self

    def transform(self, x: DataFrame) -> DataFrame:

        for col in self.variables:

            x[col] = x[col].fillna(0)

        return x


class StandardScaler(BaseEstimator, TransformerMixin):

    """
    z-score normalisation of input variables.

    Fit method stores the mean and standard deviation of input fields.

    Transform method performs calculation
    """

    def __init__(self, variables: List[str]):

        self.encoder_dict_ = {}
        self.variables = variables

    def fit(self, x: DataFrame, y=None) -> Dict:

        for col in self.variables:

            self.encoder_dict_[col] = {}

            self.encoder_dict_[col]['mean'] = x[col].mean()

            self.encoder_dict_[col]['std_dev'] = x[col].std()

        return self

    def transform(self, x: DataFrame) -> DataFrame:

        for col in self.variables:

            x[col] = (x[col] - self.encoder_dict_[col]['mean']) / self.encoder_dict_[col]['std_dev']

        return x


class ConvertToMatrix(BaseEstimator, TransformerMixin):

    """
    Converts pandas dataframe to numpy matrix to be compatible with sklearn
    """

    def fit(self, x, y=None):

        return self

    def transform(self, x: DataFrame) -> ndarray:

        x = x.values

        return x


class ReplaceValueWithNull(BaseEstimator, TransformerMixin):

    """
    Fills value with null for specified columns
    """

    def __init__(self, variables: List[str] = None, val_to_be_replaced=0):

        self.variables = variables
        self.val_to_be_replaced = val_to_be_replaced

    def fit(self, x, y=None):

        return self

    def transform(self, x: DataFrame) -> DataFrame:

        for col in self.variables:

            x[col] = x[col].replace({self.val_to_be_replaced: np.nan})

        return x


class Rounder(BaseEstimator, TransformerMixin):

    """
    Rounds columns
    """

    def __init__(self, variables: List[str] = None, decimal_places: int = 7):

        self.variables = variables
        self.decimal_places = decimal_places

    def fit(self, x, y=None):

        return self

    def transform(self, x: DataFrame) -> DataFrame:

        for col in self.variables:

            x[col] = np.round(x[col], decimals=self.decimal_places)

        return x


class CapFeatures(BaseEstimator, TransformerMixin):
    """
    Caps values between an upper and lower quantile level
    """

    def __init__(self, variables: List[str], lower_bound: float = 0.005, upper_bound: float = 0.995):

        self.cap_dict_ = {}
        self.variables = variables
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def fit(self, x: DataFrame, y=None) -> Dict:

        for col in self.variables:
            self.cap_dict_[col] = {}

            self.cap_dict_[col]['lower_cap'] = x[col].quantile(self.lower_bound)

            self.cap_dict_[col]['upper_cap'] = x[col].quantile(self.upper_bound)

        return self

    def transform(self, x: DataFrame) -> DataFrame:

        for col in self.variables:
            x[col] = np.where(x[col] < self.cap_dict_[col]['lower_cap'], self.cap_dict_[col]['lower_cap'],
                              np.where(x[col] > self.cap_dict_[col]['upper_cap'], self.cap_dict_[col]['upper_cap'],
                                       x[col]))

        return x


class Binerise(BaseEstimator, TransformerMixin):

    """
    Turns fields into binary
    """

    def __init__(self, variables: List[str] = None, threshold: int = 0):

        self.variables = variables
        self.threshold = threshold

    def fit(self, x, y=None):

        return self

    def transform(self, x: DataFrame) -> DataFrame:

        for col in self.variables:

            x[col] = np.where(x[col] > self.threshold, 1, 0)

        return x


class LowerCaseStrings(BaseEstimator, TransformerMixin):

    def __init__(self, variables: List = None):

        self.variables = variables

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        for col in self.variables:
            x[col] = x[col].str.lower()

        return x


class FillCategoricalNulls(BaseEstimator, TransformerMixin):

    def __init__(self, variables: List = None):
        self.variables = variables

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        for col in self.variables:
            x[col] = x[col].fillna('missing')

        return x


class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Rare label categorical encoder.
    Used to reduce the amount of levels in categorical field if proportion is under certain amount
    """

    def __init__(self, variables: List, tot: float = 0.01):

        self.encoder_dict_ = {}
        self.tot = tot
        self.variables = variables

    def fit(self, x, y=None):

        for col in self.variables:

            t = pd.Series(x[col].value_counts(normalize=True))

            t = t[t >= self.tot].index

            t = list(t)

            self.encoder_dict_[col] = t

        return self

    def transform(self, x):

        for col in self.variables:
            x[col] = np.where(x[col].isin(self.encoder_dict_[col]), x[col],
                              np.where(x[col].isnull(), x[col], 'rare'))

        return x


class PandasLabelEncoding(BaseEstimator, TransformerMixin):

    def __init__(self, variables: List):

        self.encoder_dict_ = {}
        self.variables = variables

    def fit(self, x, y=None):

        for col in self.variables:

            vals_dict = {}

            unique_vals = x[col].unique()

            for ind, val in enumerate(unique_vals):
                vals_dict[val] = ind

            self.encoder_dict_[col] = vals_dict

        return self

    def transform(self, x):

        for col in self.variables:
            x[col] = x[col].map(self.encoder_dict_[col])

        return x




