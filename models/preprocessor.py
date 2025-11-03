import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings

warnings.filterwarnings('ignore')


class CarPricePreprocessor:
    def __init__(self):
        self.brand_encoder = LabelEncoder()
        self.city_encoder = LabelEncoder()
        self._is_fitted = False
        self.medians = {}

    def preprocess(self, df, is_training=True):
        df_processed = df.copy()

        df_processed['brand'] = df_processed['name'].str.split().str[0]

        df_processed = self._handle_missing_values(df_processed, is_training)

        df_processed = self._encode_categorical_features(df_processed, is_training)

        df_processed = self._create_features(df_processed)

        if is_training:
            df_processed = self._remove_outliers(df_processed)
            self._is_fitted = True

        return df_processed

    def _handle_missing_values(self, df, is_training):
        df_processed = df.copy()

        numerical_features = ['year', 'mileage', 'engine_volume', 'horse_power']

        if is_training:
            for feature in numerical_features:
                if df_processed[feature].isnull().all():
                    if feature == 'year':
                        self.medians[feature] = 2020
                    elif feature == 'mileage':
                        self.medians[feature] = 50000
                    elif feature == 'engine_volume':
                        self.medians[feature] = 2.0
                    elif feature == 'horse_power':
                        self.medians[feature] = 150
                else:
                    self.medians[feature] = df_processed[feature].median()

                df_processed[feature].fillna(self.medians[feature], inplace=True)
        else:
            for feature in numerical_features:
                df_processed[feature].fillna(self.medians.get(feature, 0), inplace=True)

        df_processed['city'].fillna('Москва', inplace=True)
        df_processed['brand'].fillna('Unknown', inplace=True)

        return df_processed

    def _encode_categorical_features(self, df, is_training):
        df_processed = df.copy()

        if is_training:
            df_processed['brand_encoded'] = self.brand_encoder.fit_transform(df_processed['brand'])
            df_processed['city_encoded'] = self.city_encoder.fit_transform(df_processed['city'])
        else:
            df_processed['brand_encoded'] = self._safe_transform(self.brand_encoder, df_processed['brand'])
            df_processed['city_encoded'] = self._safe_transform(self.city_encoder, df_processed['city'])

        return df_processed

    def _safe_transform(self, encoder, series):
        result = []
        for item in series:
            if item in encoder.classes_:
                result.append(encoder.transform([item])[0])
            else:
                result.append(-1)
        return result

    def _create_features(self, df):
        df_processed = df.copy()

        current_year = 2025
        df_processed['car_age'] = current_year - df_processed['year']

        df_processed['car_age'] = df_processed['car_age'].clip(lower=0)

        df_processed['power_per_liter'] = np.where(
            df_processed['engine_volume'] > 0,
            df_processed['horse_power'] / df_processed['engine_volume'],
            0
        )

        df_processed['km_per_year'] = np.where(
            df_processed['car_age'] > 0,
            df_processed['mileage'] / df_processed['car_age'],
            df_processed['mileage']
        )

        df_processed['power_per_liter'] = df_processed['power_per_liter'].replace([np.inf, -np.inf], 0)
        df_processed['km_per_year'] = df_processed['km_per_year'].replace([np.inf, -np.inf], 0)

        df_processed['power_per_liter'].fillna(0, inplace=True)
        df_processed['km_per_year'].fillna(0, inplace=True)
        df_processed['car_age'].fillna(0, inplace=True)

        return df_processed

    def _remove_outliers(self, df):
        Q1 = df['price'].quantile(0.01)
        Q3 = df['price'].quantile(0.99)
        return df[(df['price'] >= Q1) & (df['price'] <= Q3)]

    def get_feature_names(self):
        return ['year', 'mileage', 'engine_volume', 'horse_power', 'car_age',
                'power_per_liter', 'km_per_year', 'brand_encoded', 'city_encoded']

    def prepare_features(self, df):
        processed_df = self.preprocess(df, is_training=False)
        features = processed_df[self.get_feature_names()]

        features = features.fillna(0)

        return features

    def save(self, filepath):
        joblib.dump(self, filepath)
