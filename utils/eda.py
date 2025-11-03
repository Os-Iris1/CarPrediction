import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class CarDataEDA:
    def __init__(self, df):
        self.df = df

    def basic_info(self):
        print(f"Размер датасета: {self.df.shape}")
        print(f"\nТипы данных:")
        print(self.df.dtypes)
        print(f"\nПропуски:")
        print(self.df.isnull().sum())
        print(f"\nСтатистика цен:")
        print(f"Мин: {self.df['price'].min():,} руб")
        print(f"Макс: {self.df['price'].max():,} руб")
        print(f"Медиана: {self.df['price'].median():,} руб")
        print(f"Среднее: {self.df['price'].mean():,} руб")

    def plot_price_distribution(self):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.hist(self.df['price'], bins=50, edgecolor='black', alpha=0.7)
        plt.title('Распределение цен автомобилей')
        plt.xlabel('Цена (руб)')
        plt.ylabel('Количество')

        plt.subplot(1, 2, 2)
        plt.hist(np.log1p(self.df['price']), bins=50, edgecolor='black', alpha=0.7)
        plt.title('Распределение логарифма цен')
        plt.xlabel('log(Цена)')
        plt.ylabel('Количество')

        plt.tight_layout()
        plt.show()

    def plot_mileage_vs_year(self):
        plot_data = self.df[(self.df['year'].notna()) &
                            (self.df['mileage'].notna()) &
                            (self.df['mileage'] < 500000)]

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(plot_data['year'], plot_data['mileage'],
                              c=plot_data['price'], cmap='viridis',
                              alpha=0.6, s=50)
        plt.colorbar(scatter, label='Цена (руб)')
        plt.title('Зависимость пробега от года выпуска')
        plt.xlabel('Год выпуска')
        plt.ylabel('Пробег (км)')
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_correlation_matrix(self):
        numerical_cols = ['price', 'year', 'mileage', 'engine_volume', 'horse_power']
        corr_matrix = self.df[numerical_cols].corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    fmt='.2f', square=True)
        plt.title('Матрица корреляций числовых признаков')
        plt.tight_layout()
        plt.show()

    def brand_analysis(self):
        self.df['brand'] = self.df['name'].str.split().str[0]

        top_brands = self.df['brand'].value_counts().head(10)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        top_brands.plot(kind='bar')
        plt.title('Топ-10 самых частых марок')
        plt.xlabel('Марка')
        plt.ylabel('Количество')
        plt.xticks(rotation=45)

        plt.subplot(1, 2, 2)
        brand_prices = self.df.groupby('brand')['price'].median().sort_values(ascending=False).head(10)
        brand_prices.plot(kind='bar')
        plt.title('Топ-10 самых дорогих марок (медиана)')
        plt.xlabel('Марка')
        plt.ylabel('Медианная цена (руб)')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def run_full_analysis(self):
        self.basic_info()
        self.plot_price_distribution()
        self.plot_mileage_vs_year()
        self.plot_correlation_matrix()
        self.brand_analysis()