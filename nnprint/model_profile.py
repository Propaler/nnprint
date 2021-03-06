import numpy as np
import torch.nn as nn

from tensorflow import keras
from scipy.spatial import distance
from collections import OrderedDict


class ModelProfile:
    """
    Class to map layers from a gived model and add attributes
    """

    def __init__(self, model):

        self._model = model
        self._model_type = "torch" if isinstance(model, nn.Module) else "keras"
        self._layers_info = self.fill_layers_info()

    def fill_layers_info(self):
        """
        Function to fill a dict with informations of each layer
        """
        layers_info_init = None
        # check the type of the model
        if self._model_type == "torch":
            layers_info_init = self.fill_torch()
        elif self._model_type == "keras":
            layers_info_init = self.fill_keras()
        else:
            raise NotImplementedError(
                f"There is no support for {self._model_type} framework."
            )

        return layers_info_init

    def fill_torch(self):
        """
        Fill when is a torch model
        """
        layers_info_init = OrderedDict()

        layer_names = list(list(self._model.modules())[0]._modules.keys())
        name_counter = 0
        for m in list(self._model.modules())[1:]:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                weight_copy = m.weight.data.clone().numpy()

                min_weights = np.min(weight_copy)
                max_weights = np.max(weight_copy)
                sd_weights = np.std(weight_copy)
                mean_weights = np.mean(weight_copy)

                # norms
                collaped_axis = (1, 2, 3) if isinstance(m, nn.Conv2d) else (1,)
                l1 = np.linalg.norm(weight_copy, ord=1, axis=collaped_axis)
                l2 = np.linalg.norm(weight_copy, ord=2, axis=collaped_axis)

                if isinstance(m, nn.Conv2D):
                    weight_copy = weight_copy.reshape(weight_copy.shape[0], -1)

                similarity_matrix = distance.cdist(
                    weight_copy, weight_copy, "euclidean"
                )
                gm = np.sum(np.abs(similarity_matrix), axis=0)

                layers_info_init[layer_names[name_counter]] = {
                    "sd": sd_weights,
                    "mean": mean_weights,
                    "min": min_weights,
                    "max": max_weights,
                    "norm": {"l1": l1, "l2": l2, "gm": gm},
                }

            name_counter += 1

        return layers_info_init

    def fill_keras(self):
        """
        Fill when is a keras model
        """

        layers_info_init = OrderedDict()

        for layer in self._model.layers:
            if isinstance(layer, keras.layers.Dense) or isinstance(
                layer, keras.layers.Conv2D
            ):
                weight_copy = layer.get_weights()[0]

                # Calculate
                # mean, sd, max, min
                # key is layer.name

                min_weights = np.min(weight_copy)
                max_weights = np.max(weight_copy)
                sd = np.std(weight_copy)
                mean = np.mean(weight_copy)

                # norms
                collaped_axis = (
                    (1, 2, 3) if isinstance(layer, keras.layers.Conv2D) else (1,)
                )
                l1 = np.linalg.norm(weight_copy, ord=1, exis=collaped_axis)
                l2 = np.linalg.norm(weight_copy, ord=2, exis=collaped_axis)

                layers_info_init[layer.name] = {
                    "sd": sd,
                    "mean": mean,
                    "min": min_weights,
                    "max": max_weights,
                    "norm": {"l1": l1, "l2": l2},
                }

        return layers_info_init

    def get_layers_info(self):
        """
        Return layers info
        """
        return self._layers_info

    def get_model_type(self):
        """
        Get the model's type
        """
        return self.model_type
