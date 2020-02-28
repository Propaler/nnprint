import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from tensorflow import keras

import random
import numpy as np

from PIL import Image, ImageDraw
import warnings

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
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

def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    return model


def create_whiteboard(shape=(600, 400), color="white"):
    return Image.new("RGB", shape, color)


def draw_square(
    base, topleft=(16, 16), size=16, fill="red", outline="black", width=1, padding=1
):
    """TODO add docs
    """
    draw = ImageDraw.Draw(base)
    draw.rectangle(
        [topleft, topleft[0] + size, topleft[1] + size], fill, outline, width
    )
    del draw
    return (topleft[0] + size + padding, topleft[1] + size + padding)

def draw_text(base, topleft, text):
    draw = ImageDraw.Draw(base)
    hmargin = 10
    textsize = draw.multiline_textsize(text)
    draw.multiline_text((topleft[0] - textsize[0] - hmargin, topleft[1]), text, fill="black")
    del draw

def color_palette(n):
    """Generate n random distinct colors"""
    colors = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r = int(r + step) % 256
        g = int(g + step) % 256
        b = int(b + step) % 256
        colors.append((r, g, b))
    
    return colors


def map_to_color(numpy_list):
    mn = np.amin(numpy_list)
    mx = np.amax(numpy_list)
    rescale = (numpy_list - mn) * (255/(mx - mn))
    return np.around(rescale).astype(int)

def nnprint(model, save_path="vis01.png"):
    """TODO add support to custom parameters like visualization type,
        size, and output file.
    """
    base = create_whiteboard()

    if isinstance(model, nn.Module):

        warnings.warn(
            "loading torch model.",
            Warning,
        )
        # TODO make this a class who can profile the model
        # and create a dict/map/gragh of its params.
        # An Idea is to create a structure where any information about
        # filters, norms, min, max, mean, std etc. can be obtained easily.
        square_size = 16
        padding = 1
        linewidth = 1
        extra_padding_bottom = 10
        max_line_squares = 16

        colors = color_palette(256)

        initial_point = (100, 16)
        cur_point = initial_point # TODO must be defined by default or params values

        layer_names = list(list(model.modules())[0]._modules.keys())
        # print(layer_names)

        layer_id = 0
        for m in list(model.modules())[1:]:
            if isinstance(m, nn.Conv2d):
                out_features = m.weight.data.shape[0]
                weight_copy = m.weight.data.abs().clone().numpy()
                norm = np.sum(weight_copy, axis=(1, 2, 3))
                norm_map = map_to_color(norm)

                draw_text(base, cur_point, layer_names[layer_id])
                for i in range(out_features):
                    if i > 0 and i % max_line_squares == 0:
                        cur_point = (cur_point[0] - (square_size + padding + linewidth) * max_line_squares, cur_point[1] + square_size + padding + linewidth)
                    colour = colors[norm_map[i] % 256]
                    cur_point = draw_square(base, cur_point, fill=colour)
                    cur_point = (cur_point[0] + padding, cur_point[1] - square_size - padding)
            elif isinstance(m, nn.Linear):
                out_features = m.weight.data.shape[0]
                weight_copy = m.weight.data.abs().clone().numpy()
                norm = np.sum(weight_copy, axis=(1))
                norm_map = map_to_color(norm)

                draw_text(base, cur_point, layer_names[layer_id])
                for i in range(out_features):
                    if i > 0 and i % max_line_squares == 0:
                        cur_point = (cur_point[0] - (square_size + padding + linewidth) * max_line_squares, cur_point[1] + square_size + padding + linewidth)
                    colour = colors[norm_map[i] % 256]
                    cur_point = draw_square(base, cur_point, fill=colour)
                    cur_point = (cur_point[0] + padding, cur_point[1] - square_size - padding)



            # add extra padding between layers
            cur_point = (initial_point[0], cur_point[1] + square_size + padding + linewidth + extra_padding_bottom)
            layer_id += 1

        base.save(save_path)

        return base

    elif(isinstance(model,tf.keras.Model)):
        warnings.warn(
            "loading keras model.",
            Warning,
        )

        square_size = 16
        padding = 1
        linewidth = 1
        extra_padding_bottom = 10
        max_line_squares = 16

        colors = color_palette(256)

        initial_point = (100, 16)
        cur_point = initial_point 

        layer_names = [i.name for i in model.layers]

        layer_id = 0

        for layer in model.layers:
            if isinstance(layer, keras.layers.Dense):
                out_features = layer.units
                weight_copy = np.absolute(layer.get_weights()[0])
                norm = np.sum(weight_copy, axis=(1))
                norm_map = map_to_color(norm)
                
                draw_text(base, cur_point, layer_names[layer_id])
                for i in range(out_features):
                    if i > 0 and i % max_line_squares == 0:
                        cur_point = (cur_point[0] - (square_size + padding + linewidth) * max_line_squares, cur_point[1] + square_size + padding + linewidth)
                    colour = colors[norm_map[i] % 256]
                    cur_point = draw_square(base, cur_point, fill=colour)
                    cur_point = (cur_point[0] + padding, cur_point[1] - square_size - padding)
            

            # add extra padding between layers
            cur_point = (initial_point[0], cur_point[1] + square_size + padding + linewidth + extra_padding_bottom)
            layer_id += 1

        base.save(save_path)

        return base
    else:
        print("Type model supported yet ")


# ------- TEST -------


# bottomright = (16, 16)
# for i in range(5):
#     bottomright = draw_square(base, bottomright)
#     bottomright = (bottomright[0] + 1, bottomright[1] - 16 - 1)


if __name__ == "__main__":
    model_tf = LeNet()
    model2 = create_model()
    list_created = [i.name for i in model2.layers]
    print(list_created)
    # print(model2.summary())
    
    # print(list_created)

    # nnprint(model_tf, "../images/test.png")
    nnprint(model2, "../images/test.png")

