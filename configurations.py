config = {
    'data': {
        'number_of_classes': 3,  # число классов
        'number_of_elements': 90,  # число элементов в классе
        'high': 5,  # максимальное значение центра одного класса
        'sd': 0.5,  # стандартное отклонение для генерации точек
        'data_filename': 'knn_data_logs.csv'
    },
    'plotting': {
        'cmap': 'rainbow_r',  # цветовая карта
        'grid_step': 0.05,  # дискретизация фоновой сетки
        'font_size': 'large',  # размер шрифта
        'bg_color': 'black',   # цвет фона текста
        'color': 'white',   # цвет текста
        'text_box_width': 4,   # ширина блока с текстом
        'text_box_height': 1,   # высота блока с текстом
        'figsize': (9, 6),   # размер окна
        'subplot.right': 0.7,   # положение графика в окне
        'subplot.bottom': 0.2,   # положение графика в окне
        'subplot.top': 0.9  # положение графика в окне
    },
    'knn_params': {
            'metric': 'euclidean',   # метрика
            'n_neighbors': 4,   # число соседей
            'weights': 'distance',   # тип веса
            'params_filename': 'knn_params'
    },
    'test_size': 0.2,   # размер тестовой выборки
    'make_grid':  True  # отрисовывать ли фон с классификацией

}
