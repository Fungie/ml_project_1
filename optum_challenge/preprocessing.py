from typing import List, Dict
import pandas as pd
import numpy as np
from pandas import DataFrame
from io import StringIO


class DataReader:

    """
    Used for reading in data, setting schema and removing errors associated with reading data
    """

    df_schema = {
        'id': str,
        'amount_tsh': str,
        'date_recorded': str,
        'funder': str,
        'gps_height': float,
        'installer': str,
        'longitude': float,
        'latitude': float,
        'wpt_name': str,
        'num_private': int,
        'basin': str,
        'subvillage': str,
        'region': str,
        'region_code': str,
        'district_code': str,
        'lga': str,
        'ward': str,
        'population': int,
        'public_meeting': str,
        'recorded_by': str,
        'scheme_management': str,
        'scheme_name': str,
        'permit': str,
        'construction_year': int,
        'extraction_type': str,
        'extraction_type_group': str,
        'extraction_type_class': str,
        'management': str,
        'management_group': str,
        'payment': str,
        'payment_type': str,
        'water_quality': str,
        'quality_group': str,
        'quantity': str,
        'quantity_group': str,
        'source': str,
        'source_type': str,
        'source_class': str,
        'waterpoint_type': str,
        'waterpoint_type_group': str
    }

    def __init__(self, data_loc: str = None, deploy: bool = False, api_string: str = None):

        self.data_loc = data_loc
        self.deploy = deploy
        self.api_string = api_string

        if self.deploy:

            self.df = pd.read_csv(StringIO(self.api_string), header=None, dtype=self.df_schema, names=(self.df_schema.keys()))

        else:

            self.df = pd.read_csv(self.data_loc, dtype=self.df_schema)

    def fix_amount_tsh_dtype(self) -> DataFrame:

        """
        The field 'amount_tsh' couldn't be read in as a integer due to some strings in the column.
        There were very few values less than 1 and with the scale and type of feature it would probably be hard to measure
        a value of 1. Therefore, if value is less than 1, it's set to zero

        :return: self.df: DataFrame, input dataframe with field amount_tsh fixed and dtype int
        """

        self.df.amount_tsh = np.where(self.df.amount_tsh == 'zero', '0', self.df.amount_tsh)
        self.df.amount_tsh = self.df.amount_tsh.astype(float)
        self.df.amount_tsh = np.where(self.df.amount_tsh < 1, 0, self.df.amount_tsh)
        self.df.amount_tsh = self.df.amount_tsh.astype(int)

        return self.df

    def convert_to_date(self) -> DataFrame:

        """
        Converts date recorded to date type

        :return: self.df: DataFrame, input dataframe with converted date type
        """

        self.df.date_recorded = pd.to_datetime(self.df.date_recorded)

        return self.df

    def clean_read(self):

        """
        Removes and modifies some
        :return:
        """

        self.df = self.fix_amount_tsh_dtype()
        self.df = self.convert_to_date()

        return self.df


class LabelReader:

    label_schema = {
        'id': str,
        'status_group': str
    }

    def __init__(self, label_loc):

        self.label_loc = label_loc
        self.label_df = pd.read_csv(self.label_loc, dtype=self.label_schema)
        self.label_df_clean = self.label_df.copy()

    def combine_non_functional(self) -> DataFrame:

        """
        There is a space missing in one of the labels. This method repairs this.
        :return label_df: DataFrame, label df with fixed target field
        """

        self.label_df_clean.status_group = np.where(self.label_df_clean.status_group == 'nonfunctional',
                                                    'non functional',
                                                    self.label_df_clean.status_group)

        return self.label_df_clean

    def clean_read(self) -> DataFrame:

        """
        There are some duplicate id entries, these are removed.
        :return:
        """

        self.label_df_clean = self.label_df_clean.drop_duplicates()

        self.combine_non_functional()

        return self.label_df_clean


class CombinedReader:

    def __init__(self, data_loc, label_loc):

        self.data_loc = data_loc
        self.label_loc = label_loc

    @staticmethod
    def ensure_clean_join(df: DataFrame, label_df: DataFrame) -> DataFrame:

        """
        Joins features to labels for modelling ensuring data quality

        :param df: DataFrame, features data
        :param label_df: DataFrame, labels
        :return:
        """

        df = df.drop_duplicates(subset='id', keep=False)

        df_size_before_join = df.shape[0]

        df_comb = df.merge(label_df, on='id', how='inner')

        df_size_after_join = df_comb.shape[0]

        # There were 2 labels we didn't have data for and are lost when inner joined
        assert df_size_before_join == df_size_after_join + 2

        return df_comb

    def clean_read(self):

        """
        Reads in combined features and labels with all cleaning from before.

        :return df_comb: DataFrame, combined features and target data
        """

        data_reader = DataReader(self.data_loc)
        label_reader = LabelReader(self.label_loc)

        df = data_reader.clean_read()
        label_df = label_reader.clean_read()

        df_comb = self.ensure_clean_join(df=df, label_df=label_df)

        return df_comb


