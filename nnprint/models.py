""" Toy models to make examples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import AveragePooling2D


class ThLeNet(nn.Module):
    def __init__(self):
        super(ThLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5, 5))
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class TFLeNet:
    def __init__(self):
        self._model = models.Sequential(
            [
                Conv2D(
                    6,
                    (5, 5),
                    activation="tanh",
                    input_shape=(28, 28, 1),
                    padding="same",
                ),
                Conv2D(16, (5, 5), activation="tanh", padding="valid"),
                Flatten(),
                Dense(120, activation="relu"),
                Dense(84, activation="relu"),
                Dense(10, activation="relu"),
            ]
        )

    def model(self):
        return self._model
