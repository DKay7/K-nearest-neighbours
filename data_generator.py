from numpy import random
import numpy as np
import configurations
import pandas as pd
import os.path


class DataGenerator:
    """
    Класс, генерирующий обучающую и тестовую выборки.
    """
    def __init__(self, number_of_classes, number_of_elements, high, sd, filename):
        """
        Создает объект класса

        :param int number_of_classes: Число желаемых классов
        :param int number_of_elements: Число объектов в одном классе
        :param int high: максимальная координата центра группы точек по двум осяс
        :param sd: нормальное отклонение
        :param string filename: имя файла, в который будут сохранены данные
        """

        self.number_of_classes = number_of_classes
        self.number_of_elements = number_of_elements

        self.high = high

        self.sd = sd

        self.feature_data = np.array([]).reshape(0, 2)
        self.labels = np.array([])

        self.data = None

        if os.path.isfile(filename):
            self.data = pd.read_csv(filename)

        self.generate()

    def generate(self):
        """
        Метод, генерирующий данные.
        """
        if self.data is None:
            for class_num in range(self.number_of_classes):
                x_center, y_center = random.random(2)*self.high

                for row_num in range(self.number_of_elements):
                    x = random.normal(x_center, self.sd)
                    y = random.normal(y_center, self.sd)
                    self.feature_data = np.append(self.feature_data, [[x, y]], axis=0)
                    self.labels = np.append(self.labels, class_num)

            self.data = pd.DataFrame({
                'x': self.feature_data[:, 0],
                'y': self.feature_data[:, 1],
                'labels': self.labels,
            })

            self.data.to_csv(configurations.config['data']['data_filename'], sep=',')

        else:
            x_data = self.data['x'].to_numpy()
            y_data = self.data['y'].to_numpy()
            self.feature_data = np.concatenate([[x_data, y_data]], axis=0).transpose()

            self.labels = self.data['labels'].to_numpy()

    def get_data(self):

        """
        Возвращает данные.

        :rtype: (numpy.ndarray, numpy.ndarray)
        :return: Массив координат, массив классов
        """
        return self.feature_data, self.labels

    def get_grid(self):
        """
        Метод, генерирующий матричную сетку для отрисовки фона.

        :return np.ndarray: Матрица координат
        """
        step = configurations.config['plotting']['grid_step']
        x_min, x_max = self.feature_data[:, 0].min() - 1, self.feature_data[:, 0].max() + 1
        y_min, y_max = self.feature_data[:, 1].min() - 1, self.feature_data[:, 1].max() + 1
        return np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
