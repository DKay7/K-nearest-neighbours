import matplotlib
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import os.path

import configurations


class Knn:
    """
    Класс knn, основанный на sklearn.
    """
    def __init__(self, features, labels):
        """
        Создает объект класса

        :param numpy.ndarray features: обучающая выборка
        :param numpy.ndarray labels: классы для обучающей выборки
        """
        self.cnf = configurations.config
        self.test_size = self.cnf['test_size']
        self.number_of_elements = self.cnf['data']['number_of_elements']

        self.params = self.cnf['knn_params']
        self.params_filename = self.cnf['knn_params']['params_filename']

        self.feature_matrix, self.labels = features, labels

        self.train_feature_matrix, self.test_feature_matrix, self.train_labels, self.test_labels = \
            train_test_split(self.feature_matrix, self.labels, test_size=self.test_size)

        self.knn_model = KNeighborsClassifier()

        self.params_data = None

        if os.path.isfile(self.params_filename):
            self.params_data = pd.read_csv(self.params_filename)

    def best_params(self):
        """
        Функция, подбирающая наилучшие гиперпараметры для knn через gridSearchCv.
        :return dict: наилучшие гиперпараметры
        """
        if self.params_data is None:
            all_params = {
                'metric': ['euclidean', 'manhattan', 'minkowski'],
                'n_neighbors': range(1, self.number_of_elements-1),
                'weights': ['distance', 'uniform'],
                'p': range(1, 10)
            }

            grid_search = GridSearchCV(self.knn_model, all_params, cv=5, scoring='accuracy', n_jobs=-1)

            grid_search.fit(self.train_feature_matrix, self.train_labels)

            self.params = grid_search.best_params_
            self.params_data = pd.DataFrame(self.params, index=['value'])
            self.params_data.to_csv(self.params_filename, sep=',', index=False)
            print(self.params_data)

        else:
            self.params = self.params_data.to_dict(orient='records')[0]
            print(self.params_data)
            print(self.params)

    def train(self):
        """
        Фунция, обучаяющая модель knn.
        """
        self.knn_model.set_params(**self.params)
        self.knn_model.fit(self.train_feature_matrix, self.train_labels)

    def accuracy(self):
        """
        Функция, считающая точность модели.

        :return float: точность модели
        """
        predict_labels = self.knn_model.predict(self.test_feature_matrix)
        return accuracy_score(predict_labels, self.test_labels)

    def get_accuracy(self):
        """
        Возвращает точность алгоритма

        :return float: Точность алгоритма
        """
        return self.accuracy()

    def get_params(self):
        """
        Возвращает параметры алгоритма

        :return dictionary: Точность алгоритма
        """
        return self.params

    def get_prediction(self, data=None, prob=False):
        """
        Функция, предсказывающая класс объекта.

        :param list data: данные, по которым нужно сделать предсказание,
        если None, то используется тренировочная выборка
        :param list prob: нужно ли возвращать вероятность предсказания

        :rtype: list
        :return: список предсказаний для каждого элемента (+ вероятности)
        """
        if data is None:
            data = self.test_feature_matrix
        predict_labels = self.knn_model.predict(data)

        if prob:
            predict_prob = self.knn_model.predict_proba(data)
            return predict_labels, predict_prob
        else:
            return predict_labels
