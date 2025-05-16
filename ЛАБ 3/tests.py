import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch
from io import StringIO
import sys
from AirQualityAnalyzer import AirQualityAnalyzer


class TestAirQualityAnalyzer(unittest.TestCase):
    def setUp(self):
        """Налаштування тестових даних і мокування pd.read_csv."""
        # Створюємо тестові дані
        self.test_data = pd.DataFrame({
            'country': ['Ukraine'],
            'location_name': ['Kyiv'],
            'last_updated': ['2024-05-16 11:45'],
            'air_quality_PM2.5': [2.0],
            'air_quality_PM10': [2.3],
            'temperature_celsius': [13.8],
            'humidity': [53],
            'wind_kph': [13.0],
            'air_quality_Carbon_Monoxide': [213.6],
            'air_quality_Ozone': [93.0],
            'air_quality_Nitrogen_dioxide': [0.9],
            'air_quality_Sulphur_dioxide': [2.2]
        })
        self.test_data['last_updated'] = pd.to_datetime(self.test_data['last_updated'])

        # Мокуємо pd.read_csv
        self.patcher = patch('pandas.read_csv', return_value=self.test_data)
        self.patcher.start()

        # Ініціалізуємо аналізатор
        self.analyzer = AirQualityAnalyzer()

    def tearDown(self):
        """Зупинка патчера після кожного тесту."""
        self.patcher.stop()

    def test_load_and_preprocess_data(self):
        """Перевірка завантаження та попередньої обробки даних."""
        self.assertIsNotNone(self.analyzer.data, "Дані не завантажено")
        self.assertIn('datetime', self.analyzer.data.columns, "Стовпець datetime відсутній")
        self.assertIn('date', self.analyzer.data.columns, "Стовпець date відсутній")
        self.assertIn('hour', self.analyzer.data.columns, "Стовпець hour відсутній")
        self.assertIn('month', self.analyzer.data.columns, "Стовпець month відсутній")
        self.assertIn('year', self.analyzer.data.columns, "Стовпець year відсутній")
        self.assertIn('season', self.analyzer.data.columns, "Стовпець season відсутній")
        self.assertIn('aqi', self.analyzer.data.columns, "Стовпець aqi відсутній")
        self.assertIn('air_quality_category', self.analyzer.data.columns, "Стовпець air_quality_category відсутній")

        # Перевірка значень
        self.assertEqual(self.analyzer.data.iloc[0]['season'], 'Весна', "Неправильний сезон для травня")
        self.assertEqual(self.analyzer.data.iloc[0]['month'], 5, "Неправильний місяць")
        self.assertEqual(self.analyzer.data.iloc[0]['hour'], 11, "Неправильна година")
        self.assertEqual(self.analyzer.data.iloc[0]['year'], 2024, "Неправильний рік")

    def test_calculate_aqi(self):
        """Перевірка розрахунку AQI."""
        # Тест для PM2.5=2.0, PM10=2.3
        row = self.test_data.iloc[0]
        aqi = self.analyzer.calculate_aqi(row)
        self.assertAlmostEqual(aqi, 8.33, places=2, msg="Неправильний AQI для PM2.5=2.0 та PM10=2.3")

        # Тест для PM2.5=40.0, PM10=100.0
        row = pd.Series({
            'air_quality_PM2.5': 40.0,
            'air_quality_PM10': 100.0
        })
        aqi = self.analyzer.calculate_aqi(row)
        self.assertAlmostEqual(aqi, 112.0804, places=4, msg="Неправильний AQI для PM2.5=40.0 та PM10=100.0")

    def test_air_quality_category(self):
        """Перевірка класифікації якості повітря."""
        # Перевірка для низького AQI (≈8.33)
        self.assertEqual(self.analyzer.data.iloc[0]['air_quality_category'], 'Добре',
                         "Неправильна категорія для AQI ≈ 8.33")

        # Перевірка для високого AQI
        test_data_high = self.test_data.copy()
        test_data_high['air_quality_PM2.5'] = 60.0
        test_data_high['air_quality_PM10'] = 200.0
        test_data_high['aqi'] = test_data_high.apply(self.analyzer.calculate_aqi, axis=1)
        test_data_high['air_quality_category'] = np.select(
            [
                (test_data_high['aqi'] <= 50),
                (test_data_high['aqi'] <= 100),
                (test_data_high['aqi'] <= 150),
                (test_data_high['aqi'] <= 200),
                (test_data_high['aqi'] <= 300),
                (test_data_high['aqi'] > 300)
            ],
            ['Добре', 'Помірне', 'Шкідливе для чутливих груп', 'Шкідливе', 'Дуже шкідливе', 'Небезпечне'],
            default='Невідомо'
        )
        self.assertEqual(test_data_high.iloc[0]['air_quality_category'], 'Шкідливе',
                         "Неправильна категорія для високого AQI")

    def test_get_filtered_data(self):
        """Перевірка фільтрації даних."""
        # Фільтрація за країною
        filtered = self.analyzer.get_filtered_data(country='Ukraine')
        self.assertEqual(len(filtered), 1, "Неправильна кількість записів після фільтрації за країною")
        self.assertTrue(all(filtered['country'] == 'Ukraine'), "Фільтрація за країною некоректна")

        # Фільтрація за містом
        filtered = self.analyzer.get_filtered_data(location='Kyiv')
        self.assertEqual(len(filtered), 1, "Неправильна кількість записів після фільтрації за містом")
        self.assertEqual(filtered.iloc[0]['location_name'], 'Kyiv', "Фільтрація за містом некоректна")

        # Фільтрація за датами
        filtered = self.analyzer.get_filtered_data(
            start_date='2024-05-01',
            end_date='2024-05-31'
        )
        self.assertEqual(len(filtered), 1, "Неправильна кількість записів після фільтрації за датами")
        self.assertTrue(all(filtered['date'] >= pd.to_datetime('2024-05-01').date()),
                        "Фільтрація за початковою датою некоректна")
        self.assertTrue(all(filtered['date'] <= pd.to_datetime('2024-05-31').date()),
                        "Фільтрація за кінцевою датою некоректна")

        # Фільтрація за неіснуючою країною
        filtered = self.analyzer.get_filtered_data(country='NonExistent')
        self.assertEqual(len(filtered), 0, "Фільтрація за неіснуючою країною повернула записи")

    def test_add_derived_metrics(self):
        """Перевірка додавання похідних метрик."""
        self.assertIn('year', self.analyzer.data.columns, "Стовпець year відсутній")
        self.assertEqual(self.analyzer.data.iloc[0]['year'], 2024, "Неправильний рік для 2024")

    def test_load_data_failure(self):
        """Перевірка обробки помилки при невдалому завантаженні даних."""
        # Зупиняємо попередній патчер
        self.patcher.stop()

        # Мокуємо pd.read_csv для імітації помилки
        with patch('pandas.read_csv', side_effect=FileNotFoundError("Файл не знайдено")):
            # Перенаправляємо stdout для захоплення виводу
            captured_output = StringIO()
            sys.stdout = captured_output
            try:
                analyzer = AirQualityAnalyzer()
                self.assertIsNone(analyzer.data, "Дані не повинні бути завантажені при помилці")
                self.assertIn("Помилка завантаження даних", captured_output.getvalue(),
                              "Повідомлення про помилку не виведено")
            finally:
                # Відновлюємо stdout
                sys.stdout = sys.__stdout__


if __name__ == '__main__':
    unittest.main()
