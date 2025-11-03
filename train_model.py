import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from models.model_trainer import CarPriceModelTrainer


def main():
    data_path = "data/drom_cars_dataset.csv"

    trainer = CarPriceModelTrainer()
    df = trainer.load_data(data_path)

    trainer.prepare_data()

    results = trainer.train_model()

    trainer.plot_predictions()

    trainer.save_models(
        model_path='models/best_model.joblib',
        preprocessor_path='models/preprocessor.joblib'
    )

    model_info = trainer.get_model_info()
    print(f"\nИнформация о модели:\n")
    print(f"Модель: {model_info['model_name']}")
    print(f"R^2 score: {model_info['r2_score']:.3f}")
    print(f"MAE: {model_info['mae']:,.0f} руб")
    print(f"Используемые признаки: {model_info['features']}")


if __name__ == "__main__":
    main()