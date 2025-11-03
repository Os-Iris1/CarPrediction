import pandas as pd
import numpy as np
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
from .preprocessor import CarPricePreprocessor


class CarPriceModelTrainer:
    def __init__(self):
        self.preprocessor = CarPricePreprocessor()
        self.model = None
        self.results = {}

    def load_data(self, filepath):
        self.df = pd.read_csv(filepath)
        return self.df

    def prepare_data(self):
        self.df_processed = self.preprocessor.preprocess(self.df, is_training=True)
        feature_names = self.preprocessor.get_feature_names()

        self.df_processed[feature_names] = self.df_processed[feature_names].fillna(0)

        self.X = self.df_processed[feature_names]
        self.y = self.df_processed['price']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)

        mae = mean_absolute_error(self.y_test, y_pred)
        mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
        r2 = r2_score(self.y_test, y_pred)

        self.results = {
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'predictions': y_pred
        }

        return self.results

    def plot_predictions(self):

        if not self.results:
            return

        predictions = self.results['predictions']

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(self.y_test, predictions, alpha=0.6)
        plt.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Истинная цена (руб)')
        plt.ylabel('Предсказанная цена (руб)')
        plt.title('Предсказанная vs Истинная цена')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        errors = np.abs(self.y_test - predictions)
        plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Абсолютная ошибка (руб)')
        plt.ylabel('Количество')
        plt.title('Распределение ошибок предсказания')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def save_models(self, model_path='best_model.joblib',
                    preprocessor_path='preprocessor.joblib'):
        import os
        os.makedirs('models', exist_ok=True)

        joblib.dump(self.model, model_path)
        self.preprocessor.save(preprocessor_path)

    def get_model_info(self):
        if not self.model:
            return

        return {
            'model_name': 'Random Forest',
            'r2_score': self.results['R2'],
            'mae': self.results['MAE'],
            'features': self.preprocessor.get_feature_names()
        }
