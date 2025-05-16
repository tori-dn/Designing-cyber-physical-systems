import pandas as pd
import numpy as np

class AirQualityAnalyzer:
    def __init__(self):
        # Ініціалізація з завантаженням даних
        self.data = self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        """Завантаження та попередня обробка даних із CSV-файлу"""
        try:
            # Читаємо CSV-файл із даними про погоду та якість повітря
            df = pd.read_csv('GlobalWeatherRepository.csv')  # Шлях у корені проєкту
            # Конвертація стовпця 'last_updated' у формат datetime
            df['datetime'] = pd.to_datetime(df['last_updated'])
            # Витягуємо дату, час, годину, місяць і рік із datetime
            df['date'] = df['datetime'].dt.date
            df['time'] = df['datetime'].dt.time
            df['hour'] = df['datetime'].dt.hour
            df['month'] = df['datetime'].dt.month
            df['year'] = df['datetime'].dt.year
            # Визначаємо сезон на основі місяця
            df['season'] = df['month'].apply(
                lambda m: 'Зима' if m in [12, 1, 2] else
                'Весна' if m in [3, 4, 5] else
                'Літо' if m in [6, 7, 8] else 'Осінь'
            )
            # Розраховуємо AQI (Індекс якості повітря) для кожного рядка
            df['aqi'] = df.apply(self.calculate_aqi, axis=1)
            # Класифікуємо якість повітря на основі AQI
            conditions = [
                (df['aqi'] <= 50),
                (df['aqi'] <= 100),
                (df['aqi'] <= 150),
                (df['aqi'] <= 200),
                (df['aqi'] <= 300),
                (df['aqi'] > 300)
            ]
            choices = [
                'Добре',
                'Помірне',
                'Шкідливе для чутливих груп',
                'Шкідливе',
                'Дуже шкідливе',
                'Небезпечне'
            ]
            df['air_quality_category'] = np.select(conditions, choices, default='Невідомо')
            return df
        except Exception as e:
            # Виводимо помилку, якщо дані не вдалося завантажити
            print(f"Помилка завантаження даних: {e}")
            return None

    def calculate_aqi(self, row):
        """Розрахунок AQI на основі PM2.5 і PM10"""
        pm25 = row['air_quality_PM2.5']
        pm10 = row['air_quality_PM10']

        def calculate_component_index(Cp, Ih, Il, BPh, BPl):
            """Розрахунок AQI для окремого компонента"""
            return ((Ih - Il) / (BPh - BPl)) * (Cp - BPl) + Il

        # PM2.5 breakpoints
        if pm25 <= 12.0:
            aqi_pm25 = calculate_component_index(pm25, 50, 0, 12.0, 0)
        elif pm25 <= 35.4:
            aqi_pm25 = calculate_component_index(pm25, 100, 51, 35.4, 12.1)
        elif pm25 <= 55.4:
            aqi_pm25 = calculate_component_index(pm25, 150, 101, 55.4, 35.5)
        elif pm25 <= 150.4:
            aqi_pm25 = calculate_component_index(pm25, 200, 151, 150.4, 55.5)
        elif pm25 <= 250.4:
            aqi_pm25 = calculate_component_index(pm25, 300, 201, 250.4, 150.5)
        else:
            aqi_pm25 = calculate_component_index(pm25, 500, 301, 500.4, 250.5)

        # PM10 breakpoints
        if pm10 <= 54:
            aqi_pm10 = calculate_component_index(pm10, 50, 0, 54, 0)
        elif pm10 <= 154:
            aqi_pm10 = calculate_component_index(pm10, 100, 51, 154, 55)
        elif pm10 <= 254:
            aqi_pm10 = calculate_component_index(pm10, 150, 101, 254, 155)
        elif pm10 <= 354:
            aqi_pm10 = calculate_component_index(pm10, 200, 151, 354, 255)
        elif pm10 <= 424:
            aqi_pm10 = calculate_component_index(pm10, 300, 201, 424, 355)
        else:
            aqi_pm10 = calculate_component_index(pm10, 500, 301, 604, 425)

        # Повертаємо максимальне значення AQI
        return max(aqi_pm25, aqi_pm10)

    def get_filtered_data(self, country=None, location=None, start_date=None, end_date=None):
        """Фільтрація даних за країною, містом і діапазоном дат"""
        if self.data is None:
            return pd.DataFrame()

        filtered_data = self.data.copy()

        # Фільтрація за країною
        if country:
            filtered_data = filtered_data[filtered_data['country'] == country]

        # Фільтрація за містом
        if location:
            filtered_data = filtered_data[filtered_data['location_name'] == location]

        # Фільтрація за діапазоном дат
        if start_date:
            try:
                start_date = pd.to_datetime(start_date).date()
                filtered_data = filtered_data[filtered_data['date'] >= start_date]
            except ValueError:
                pass  # Ігноруємо некоректний формат дати

        if end_date:
            try:
                end_date = pd.to_datetime(end_date).date()
                filtered_data = filtered_data[filtered_data['date'] <= end_date]
            except ValueError:
                pass  # Ігноруємо некоректний формат дати

        return filtered_data
