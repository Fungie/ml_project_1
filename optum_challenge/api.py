from typing import Dict
from optum_challenge.preprocessing import DataReader
from optum_challenge.feature_engineering import FeatureEngineering
import joblib
import numpy as np


class Api:

    """
    Model prediction API on new data. Meant to mimic a real api.
    """

    def __init__(self, post: str):

        self.post = post

    def get(self) -> Dict:

        reader = DataReader(deploy=True, api_string=self.post)
        df = reader.clean_read()

        feature_maker = FeatureEngineering(df)
        df_pred = feature_maker.calculate_age()

        preprocessing_pipe = joblib.load('model_objects/data_pipeline.pkl')
        df_pred = preprocessing_pipe.transform(df_pred)

        model = joblib.load('model_objects/final_model.pkl')
        df_pred['prediction_number'] = np.argmax(model.predict(df_pred), axis=1)

        label_map = joblib.load('model_objects/label_values.pkl')
        inv_label_map = {v: k for k, v in label_map.items()}

        df_pred['prediction'] = df_pred.prediction_number.map(inv_label_map)

        df_pred = df_pred[['prediction']]
        df = df[['id']]

        df = df.join(df_pred)

        result = df.to_dict('records')

        if len(result) == 1:

            result = result[0]

        return result







