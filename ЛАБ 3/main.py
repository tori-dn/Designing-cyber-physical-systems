import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QComboBox, QDateEdit, QTextEdit, QTabWidget,
                            QMessageBox, QTableWidget, QTableWidgetItem, QLabel)
from PyQt5.QtCore import QDate
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from AirQualityAnalyzer import AirQualityAnalyzer  # Імпорт із AirQualityAnalyzer.py

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = AirQualityAnalyzer()  # Ініціалізація аналізатора
        self.initUI()

    def initUI(self):
        """Ініціалізація графічного інтерфейсу"""
        self.setWindowTitle('Аналіз якості повітря')
        self.setGeometry(100, 100, 1200, 800)

        # Центральний віджет і макет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Вкладки
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Вкладка 1: Аналітика
        self.analytics_tab = QWidget()
        analytics_layout = QVBoxLayout(self.analytics_tab)

        # Фільтри
        filters_layout = QHBoxLayout()
        self.country_combo = QComboBox()
        self.city_combo = QComboBox()
        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate.currentDate().addMonths(-1))
        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate())
        filter_button = QPushButton('Фільтрувати')
        filter_button.clicked.connect(self.update_analytics)

        filters_layout.addWidget(QLabel('Країна:'))
        filters_layout.addWidget(self.country_combo)
        filters_layout.addWidget(QLabel('Місто:'))
        filters_layout.addWidget(self.city_combo)
        filters_layout.addWidget(QLabel('Початкова дата:'))
        filters_layout.addWidget(self.start_date)
        filters_layout.addWidget(QLabel('Кінцева дата:'))
        filters_layout.addWidget(self.end_date)
        filters_layout.addWidget(filter_button)
        analytics_layout.addLayout(filters_layout)

        # Таблиця для відображення даних
        self.table = QTableWidget()
        analytics_layout.addWidget(self.table)

        # Графіки
        self.figure, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvas(self.figure)
        analytics_layout.addWidget(self.canvas)

        self.tabs.addTab(self.analytics_tab, 'Аналітика')

        # Вкладка 2: Рекомендації
        self.recommendations_tab = QWidget()
        recommendations_layout = QVBoxLayout(self.recommendations_tab)
        self.recommendations_text = QTextEdit()
        self.recommendations_text.setReadOnly(True)
        recommendations_layout.addWidget(self.recommendations_text)
        self.tabs.addTab(self.recommendations_tab, 'Рекомендації')

        # Заповнення випадаючих списків
        self.populate_filters()

    def populate_filters(self):
        """Заповнення випадаючих списків країнами та містами"""
        if self.analyzer.data is not None:
            countries = [''] + sorted(self.analyzer.data['country'].unique())
            self.country_combo.addItems(countries)
            cities = [''] + sorted(self.analyzer.data['location_name'].unique())
            self.city_combo.addItems(cities)
            self.country_combo.currentTextChanged.connect(self.update_cities)
        else:
            self.country_combo.addItem('Дані відсутні')
            self.city_combo.addItem('Дані відсутні')

    def update_cities(self):
        """Оновлення списку міст при зміні країни"""
        self.city_combo.clear()
        country = self.country_combo.currentText()
        if self.analyzer.data is not None:
            if country:
                cities = [''] + sorted(self.analyzer.data[self.analyzer.data['country'] == country]['location_name'].unique())
            else:
                cities = [''] + sorted(self.analyzer.data['location_name'].unique())
            self.city_combo.addItems(cities)
        else:
            self.city_combo.addItem('Дані відсутні')

    def update_analytics(self):
        """Оновлення таблиці, графіків і рекомендацій на основі фільтрів"""
        country = self.country_combo.currentText() if self.country_combo.currentText() else None
        city = self.city_combo.currentText() if self.city_combo.currentText() else None
        start_date = self.start_date.date().toString('yyyy-MM-dd') if self.start_date.date() else None
        end_date = self.end_date.date().toString('yyyy-MM-dd') if self.end_date.date() else None

        filtered_data = self.analyzer.get_filtered_data(
            country=country,
            location=city,
            start_date=start_date,
            end_date=end_date
        )

        # Оновлення таблиці
        self.table.setRowCount(filtered_data.shape[0])
        self.table.setColumnCount(filtered_data.shape[1])
        self.table.setHorizontalHeaderLabels(filtered_data.columns)

        for i in range(filtered_data.shape[0]):
            for j in range(filtered_data.shape[1]):
                self.table.setItem(i, j, QTableWidgetItem(str(filtered_data.iloc[i, j])))

        # Оновлення графіків
        self.ax.clear()
        if not filtered_data.empty:
            filtered_data.groupby('date')['aqi'].mean().plot(ax=self.ax)
            self.ax.set_title('Середній AQI за датами')
            self.ax.set_xlabel('Дата')
            self.ax.set_ylabel('AQI')
        self.canvas.draw()

        # Оновлення рекомендацій
        self.show_detailed_recommendations(filtered_data)

    def show_detailed_recommendations(self, data):
        """Відображення рекомендацій на основі останніх даних"""
        if data.empty:
            self.recommendations_text.setText("Немає даних для відображення рекомендацій.")
            return

        latest_data = data.sort_values('datetime').iloc[-1]
        aqi = latest_data['aqi']
        category = latest_data['air_quality_category']

        recommendations = f"Останній AQI: {aqi:.2f}\nКатегорія якості повітря: {category}\n\nРекомендації:\n"
        if category == 'Добре':
            recommendations += "- Якість повітря хороша. Приємно проводити час на вулиці!\n"
        elif category == 'Помірне':
            recommendations += "- Якість повітря прийнятна. Чутливим групам слід обмежити тривале перебування на вулиці.\n"
        elif category == 'Шкідливе для чутливих груп':
            recommendations += "- Чутливим групам (діти, літні люди, люди з респіраторними захворюваннями) слід уникати фізичних навантажень на вулиці.\n"
        elif category == 'Шкідливе':
            recommendations += "- Уникайте тривалого перебування на вулиці. Використовуйте маски при необхідності.\n"
        elif category == 'Дуже шкідливе':
            recommendations += "- Залишайтеся в приміщенні, уникайте зовнішнього повітря. Використовуйте очищувачі повітря.\n"
        elif category == 'Небезпечне':
            recommendations += "- Уникайте контакту із зовнішнім повітрям. Використовуйте респіратори та очищувачі повітря.\n"

        self.recommendations_text.setText(recommendations)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Перевірка наявності файлу даних
    if not os.path.exists('GlobalWeatherRepository.csv'):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText("Файл даних не знайдено!")
        msg.setInformativeText("Будь ласка, розмістіть GlobalWeatherRepository.csv у кореневій папці проєкту.")
        msg.exec_()
        sys.exit(1)

    # Створення та відображення головного вікна
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
