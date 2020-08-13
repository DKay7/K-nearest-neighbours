import knn
import plot
import data_generator
import configurations
import numpy as np

# plot_object = None
# data_gen = None
# knn_model = None
# features, labels = None, None

# while True:
#     if plot_object or knn_model or data_gen or features or labels is None:

plot_object = plot.Visualisation()

data_gen = data_generator.DataGenerator(*configurations.config["data"].values())
features, labels = data_gen.get_data()
knn_model = knn.Knn(features, labels)
knn_model.best_params()
knn_model.train()
print(knn_model.accuracy())

plot_object.set_data(features, labels)
plot_object.set_knn_model(knn_model)
plot_object.set_data_generator(data_gen)
plot_object.show_data(make_grid=configurations.config['make_grid'])

# plot_object.animate()
