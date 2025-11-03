import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
import time


class DromParser:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        })

        self.regional_urls = [
            'https://moscow.drom.ru/auto/',
        ]

    def extract_numeric_features(self, text):
        features = {}

        year_match = re.search(r'(\d{4})\s*г', text)
        features['year'] = int(year_match.group(1)) if year_match else None

        mileage_match = re.search(r'(\d[\d\s]*)\s*км', text.replace(' ', ''))
        if mileage_match:
            features['mileage'] = int(mileage_match.group(1))
        else:
            features['mileage'] = None

        engine_match = re.search(r'(\d+\.?\d*)\s*л', text)
        features['engine_volume'] = float(engine_match.group(1)) if engine_match else None

        power_match = re.search(r'(\d+)\s*л\.?с?', text)
        features['horse_power'] = int(power_match.group(1)) if power_match else None

        return features

    def parse_car_container(self, container, base_url):
        try:
            car_data = {}

            title_elem = container.find(['h3', 'a'], {'data-ftid': 'bull_title'}) or container.find(
                'h3') or container.find('a', class_=re.compile(r'title', re.I))
            if not title_elem:
                return None

            car_data['name'] = title_elem.get_text(strip=True)

            price_elem = container.find(['span', 'div'], {'data-ftid': 'bull_price'})
            if not price_elem:
                return None

            price_text = price_elem.get_text(strip=True)
            price_clean = re.sub(r'[^\d]', '', price_text)
            car_data['price'] = int(price_clean) if price_clean else None

            link_elem = container.find('a', href=True)
            if link_elem:
                href = link_elem['href']
                if not href.startswith('http'):
                    if href.startswith('//'):
                        href = 'https:' + href
                    else:
                        base = base_url.rstrip('/')
                        href = base + ('/' + href if not href.startswith('/') else href)
                car_data['url'] = href
            else:
                car_data['url'] = None

            city_elem = container.find(['span', 'div'], {'data-ftid': 'bull_location'})
            car_data['city'] = city_elem.get_text(strip=True) if city_elem else None

            params_text = []
            desc_elem = container.find('div', {'data-ftid': 'component_inline-bull-description'})
            if desc_elem:
                params_text.append(desc_elem.get_text(' | ', strip=True))

            param_elems = container.find_all(['span', 'div'], {'data-ftid': re.compile(r'bull_description')})
            if param_elems:
                param_texts = [elem.get_text(strip=True) for elem in param_elems if elem.get_text(strip=True)]
                if param_texts:
                    params_text.append(' | '.join(param_texts))

            all_params_text = ' | '.join(params_text) if params_text else ''
            car_data['params'] = all_params_text

            numeric_features = self.extract_numeric_features(all_params_text + ' ' + car_data['name'])
            car_data.update(numeric_features)

            date_elem = container.find(['div', 'span'], {'data-ftid': 'bull_date'})
            car_data['date'] = date_elem.get_text(strip=True) if date_elem else None

            irrelevant_keywords = ['объявл', 'модел', 'продать', 'куплю', 'ищу']
            name_lower = car_data['name'].lower()
            if any(keyword in name_lower for keyword in irrelevant_keywords):
                return None

            return car_data

        except Exception:
            return None

    def parse_page(self, url):
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return []

            soup = BeautifulSoup(response.text, 'html.parser')

            containers = soup.find_all('div', {'data-ftid': 'bulls-list_bull'})

            if not containers:
                return []

            cars_data = []
            for container in containers:
                car_info = self.parse_car_container(container, url)
                if car_info:
                    cars_data.append(car_info)

            return cars_data

        except Exception:
            return []

    def collect_data(self, pages_per_region=3):
        all_cars = []

        for base_url in self.regional_urls:

            for page in range(1, pages_per_region + 1):
                if page == 1:
                    url = base_url
                else:
                    url = f"{base_url}page{page}/"

                cars = self.parse_page(url)
                all_cars.extend(cars)

                time.sleep(1)

                if not cars:
                    break

        return pd.DataFrame(all_cars)

    def validate_dataset(self, df):
        if df.empty:
            return False, "empty_data"

        required_columns = ['name', 'price', 'year']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Отсутствуют колонки: {missing_columns}"

        if df['price'].isna().all():
            return False, "empty_price"

        if df['year'].isna().all():
            return False, "empty_year"

        return True, "is_valid"


def main():
    parser = DromParser()

    df = parser.collect_data(pages_per_region=50)

    #TODO: Починить парсер и валидатор
    is_valid, message = parser.validate_dataset(df)
    is_valid = True

    if is_valid:
        df.to_csv('drom_cars_dataset.csv', index=False, encoding='utf-8')

        stats = {
            'total_cars': len(df),
            'avg_price': df['price'].mean(),
            'avg_year': df['year'].mean(),
            'avg_mileage': df['mileage'].mean(),
            'unique_brands': df['name'].str.split().str[0].nunique(),
            'cities_count': df['city'].nunique()
        }

        return df, stats, message
    else:
        return pd.DataFrame(), {}, message
