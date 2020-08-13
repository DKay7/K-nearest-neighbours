import matplotlib.pyplot as plt
import matplotlib
import configurations
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.animation import FuncAnimation
import os

import data_generator
import knn


class Visualisation:
    """
    Класс, отвечающий за отрисовку всех объектов.
    """
    def __init__(self):
        self.config = configurations.config['plotting']
        matplotlib.rcParams["figure.figsize"] = self.config['figsize']
        matplotlib.rcParams["figure.subplot.right"] = self.config['subplot.right']
        matplotlib.rcParams["figure.subplot.bottom"] = self.config['subplot.bottom']
        matplotlib.rcParams["figure.subplot.top"] = self.config['subplot.top']
        self.cmap = self.config['cmap']
        self.fig = plt.figure()
        self.plt = plt

        self.axis = plt.axes()
        self.dot = None
        self.dot_text = None
        self.info = None

        self.num_classes = configurations.config['data']['number_of_classes']
        self.knn_model = None
        self.data_gen = None
        self.features = []
        self.labels = []

        self.refresh()

    def refresh(self):
        self.data_gen = data_generator.DataGenerator(*configurations.config["data"].values())
        self.features, self.labels = self.data_gen.get_data()

        self.knn_model = knn.Knn(self.features, self.labels)
        self.knn_model.best_params()
        self.knn_model.train()

    def set_data(self, features, labels):
        """
        Устанавливает данные обучающей выборки.

        :param numpy.ndarray features: обучающая выборка
        :param numpy.ndarray labels: классы для обучающей выборки
        """
        self.features = features
        self.labels = labels

    def set_knn_model(self, knn_m):
        """
        Устанавливает модель knn.

        :type knn_m: knn.Knn
        :param knn_m: Модель knn
        """
        self.knn_model = knn_m

    def set_data_generator(self, data_gen):
        """
        Устанавливает генератор данных.

        :type data_gen: data_generator.DataGenerator
        :param data_gen: объект генератора данных
        """
        self.data_gen = data_gen

    def show_data(self, make_grid=True):
        """
        Основная функция, отвечающая за отрисовку обучающей выборки,
        а также цветной фоновой сетки, предсказывающей класс.

        :param boolean make_grid: Отрисовывать ли фон с классификацией
        """

        if make_grid:
            xx, yy = self.data_gen.get_grid()
            predicted = self.knn_model.get_prediction(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

            self.plt.pcolormesh(xx, yy, predicted, cmap=self.cmap, alpha=1, label='Класс')

        sct = self.plt.scatter(self.features[:, 0], self.features[:, 1], c=self.labels,
                               s=60,
                               norm=plt.Normalize(vmin=0, vmax=self.num_classes-1),
                               cmap=self.cmap, edgecolors='black', linewidth=2,
                               label='class {0}'.format(1))

        self.axis.set_title('Метод K ближайших соседей\n'
                            'поставить точку - клавиша мыши,\n убрать - двойной щелчок')

        self.info = self.plt.text(self.axis.get_xlim()[1]+0.3, self.axis.get_ylim()[1] - 2.15,
                                  s='Параметры алгоритма:\n'
                                  + '\n'.join([str(key)+': '+str(self.knn_model.get_params().get(key))
                                              for key in self.knn_model.get_params().keys()])

                                  + '\n' + 'Точность алгоритма: '
                                  + str(round(self.knn_model.get_accuracy(), 3)),

                                  color=self.config['color'],
                                  bbox={"facecolor": self.config['bg_color'],
                                        "linestyle": "dotted",
                                        "linewidth": 2.0
                                        },
                                  fontsize=self.config['font_size']
                                  )

        self.fig.legend(handles=sct.legend_elements()[0],
                        labels=["Class {0}".format(i) for i in range(self.num_classes)],
                        loc='lower center',
                        ncol=self.num_classes
                        )

        self.interactive()
        self.plt.show()

    def interactive(self):
        """
        Функция, отвечающая за интерактивное взаимодействие с графиком.
        """
        _ = self.plt.connect('button_press_event', self.onclick)

    def onclick(self, event):
        """
        Функция-прослушитель, вызывается, когда нажата кнопка мышки.
        Отвечает за отрисовку точки и ее класса по клику мыши.
        :param event:
        :return:
        """
        axes = event.inaxes

        if axes is None:
            return

        _, size_x = axes.get_xlim()
        _, size_y = axes.get_ylim()

        if (self.dot or self.dot_text) is not None:
            self.dot.remove()
            self.dot_text.remove()

        x, y = event.xdata, event.ydata

        label, prob = self.knn_model.get_prediction([[x, y]], prob=True)

        self.dot = self.plt.scatter(x, y,
                                    s=60,
                                    c=int(label[0]),
                                    norm=plt.Normalize(vmin=0, vmax=self.num_classes-1),
                                    edgecolors='black',
                                    cmap=self.cmap,
                                    linewidth=2)
        self.info.set_x(size_x + 0.3)
        self.info.set_y(size_y-2.5)
        self.dot_text = plt.text(size_x + 0.3, size_y-4,
                                 'Класс {0}\nВероятность:{1}'.format(
                                     label[0],
                                     round(prob[0][int(label[0])], 3)
                                 ),
                                 color=self.config['color'],
                                 bbox={"facecolor": self.config['bg_color'],
                                       "linestyle": "dotted",
                                       "linewidth": 2.0
                                       },
                                 fontsize=self.config['font_size']
                                 )

        if event.dblclick:
            self.dot.remove()
            self.dot_text.remove()
            self.dot = None
            self.dot_text = None

        self.plt.draw()

    # def animate(self):
    #     frame = np.arange(-50.0, 200.0, 1.0)
    #     animation = FuncAnimation(
    #         self.fig,
    #         func=self.update,
    #         frames=frame,
    #     )
    #     self.plt.show()
    #
    # def update(self, frame):
    #     while True:
    #         if input() == '0':
    #             os.remove(configurations['knn_params']['params_filename'])
    #             os.remove(configurations['data']['data_filename'])
    #             self.refresh()
    #             self.show_data()
    #
    #         self.show_data()
