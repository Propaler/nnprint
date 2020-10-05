import logging
import os

import torch.nn as nn
import torch
import tensorflow as tf
from tensorflow import keras

import numpy as np
from scipy.spatial import distance

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import warnings

from nnprint import utils


logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

torch.manual_seed(41)

def create_whiteboard(shape=(600, 600), color="white"):
    return Image.new("RGB", shape, color)


def draw_title(base, point, title, title_font_size):
    """TODO add docs"""

    if title is not None:

        draw = ImageDraw.Draw(base)
        fnt = ImageFont.truetype(
            "Pillow/Tests/fonts/FreeMono.ttf", title_font_size
        )

        draw.text(point, title, font=fnt, fill="black", align="left")

        del draw


def draw_square(
    base,
    topleft=(16, 16),
    size=16,
    fill="red",
    outline="lightgrey",
    width=1,
    inner_square_margin=1,
):
    """TODO add docs"""
    draw = ImageDraw.Draw(base)
    draw.rectangle(
        [topleft, topleft[0] + size, topleft[1] + size], fill, outline, width
    )
    del draw
    return (
        topleft[0] + size + inner_square_margin,
        topleft[1] + size + inner_square_margin,
    )


def draw_text(base, topleft, text, fill="black", position="left"):
    draw = ImageDraw.Draw(base)
    hmargin = 10
    text = str(text)
    textsize = draw.multiline_textsize(text)
    offset = 0
    if position == "right":
        offset = textsize[0] + hmargin
    elif position == "left":
        offset = -(textsize[0] + hmargin)

    draw.multiline_text(
        (topleft[0] + offset, topleft[1]),
        text,
        fill=fill,
        align="left",
    )
    del draw


def color_palette(pallete="greyscale"):
    """Generate n random distinct colors"""
    # colors = []
    # r = int(random.random() * 256)
    # g = int(random.random() * 256)
    # b = int(random.random() * 256)
    # step = 256 / n
    # for i in range(n):
    #     r = int(r + step) % 256
    #     g = int(g + step) % 256
    #     b = int(b + step) % 256
    #     colors.append((r, g, b))

    colors = utils.pallete_greyscale
    if pallete == "purplescale":
        colors = utils.pallete_mix

    return colors


def map_to_color(numpy_list):
    mn = np.amin(numpy_list)
    mx = np.amax(numpy_list)
    rescale = (numpy_list - mn) * (255 / (mx - mn))
    return np.around(rescale).astype(int)


# def group_similar_colors(colors):
#     newlist = list()
#     prev, norm = colors[0]
#     thresh = 100
#     newlist.append((prev, norm))
#     for i in range(1, len(colors)):
#         r, g, b = colors[i][0]
#         pr, pg, pb = prev
#         if abs(r - pr) <= thresh and abs(g - pg) <=
#                       thresh and abs(b - pb) <= thresh:
#             continue
#         newlist.append(colors[i])
#         prev = colors[i][0]

#     return newlist


def nnprint(model, importance_criteria="l1", save_path="vis01.png", title_font_size=30, title=None, subtitle=None):
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
        inner_square_margin = 1
        linewidth = 1
        extra_padding_bottom = 10
        extra_margin_left = 30
        max_line_squares = 16
        unique_colors = set()

        colors = color_palette()
        num_colors = len(colors)

        initial_point_title = (100, 16)
        initial_point = (100, 16 + title_font_size)
        cur_point = initial_point  # TODO must be defined by default or params values

        draw_title(base, initial_point_title, title, title_font_size)
        layer_names = list(list(model.modules())[0]._modules.keys())


        layer_id = 0
        for m in list(model.modules())[1:]:
            if isinstance(m, nn.Conv2d):
                weight_copy = m.weight.data.clone().numpy()

                if importance_criteria == "l1":
                    abs_weight = np.abs(weight_copy)
                    norm = np.sum(abs_weight, axis=(1, 2, 3))
                elif importance_criteria == "l2":
                    squared_weight = np.square(weight_copy)
                    norm = np.sum(squared_weight, axis=(1, 2, 3))
                elif importance_criteria == "gm":
                    weight_copy = weight_copy.reshape(weight_copy.shape[0], -1)
                    similarity_matrix = distance.cdist(
                        weight_copy, weight_copy, "euclidean"
                    )
                    norm = np.sum(np.abs(similarity_matrix), axis=0)
                else:
                    raise NotImplementedError(
                        f"There is no support for {importance_criteria} criteria."
                    )

                norm_map = map_to_color(norm)

                draw_text(base, cur_point, layer_names[layer_id])
                out_features = m.weight.data.shape[0]

                for i in range(out_features):
                    if i > 0 and i % max_line_squares == 0:
                        cur_point = (
                            cur_point[0]
                            - (square_size + inner_square_margin + linewidth)
                            * max_line_squares,
                            cur_point[1]
                            + square_size
                            + inner_square_margin
                            + linewidth,
                        )
                    colour_index = norm_map[i] % num_colors
                    colour = colors[colour_index]
                    unique_colors.add((colour, colour_index))
                    cur_point = draw_square(base, cur_point, fill=colour)
                    cur_point = (
                        cur_point[0] + inner_square_margin,
                        cur_point[1] - square_size - inner_square_margin,
                    )
            elif isinstance(m, nn.Linear):
                weight_copy = m.weight.data.clone().numpy()

                if importance_criteria == "l1":
                    abs_weight = np.abs(weight_copy)
                    norm = np.sum(abs_weight, axis=(1))
                elif importance_criteria == "l2":
                    squared_weight = np.square(weight_copy)
                    norm = np.sum(squared_weight, axis=(1))
                elif importance_criteria == "gm":
                    similarity_matrix = distance.cdist(
                        weight_copy, weight_copy, "euclidean"
                    )
                    norm = np.sum(np.abs(similarity_matrix), axis=0)
                else:
                    raise NotImplementedError(
                        f"There is no support for {importance_criteria} criteria."
                    )

                norm_map = map_to_color(norm)

                draw_text(base, cur_point, layer_names[layer_id])
                out_features = m.weight.data.shape[0]

                for i in range(out_features):
                    if i > 0 and i % max_line_squares == 0:
                        cur_point = (
                            cur_point[0]
                            - (square_size + inner_square_margin + linewidth)
                            * max_line_squares,
                            cur_point[1]
                            + square_size
                            + inner_square_margin
                            + linewidth,
                        )
                    colour_index = norm_map[i] % num_colors
                    colour = colors[colour_index]
                    unique_colors.add((colour, colour_index))
                    cur_point = draw_square(base, cur_point, fill=colour)
                    cur_point = (
                        cur_point[0] + inner_square_margin,
                        cur_point[1] - square_size - inner_square_margin,
                    )

            # add extra padding between layers
            cur_point = (
                initial_point[0],
                cur_point[1]
                + square_size
                + inner_square_margin
                + linewidth
                + extra_padding_bottom,
            )
            layer_id += 1

        # adding legend
        # FIXME should be possible set legend position

        legend_colors = list(unique_colors)

        # sort by norm
        legend_colors.sort(key=lambda item: item[1])

        # print(legend_colors)
        cur_point = (
            initial_point[0]
            + square_size * (max_line_squares + linewidth + inner_square_margin)
            + extra_margin_left,
            initial_point[1],
        )
        for i, (colour, aprox_norm) in enumerate(legend_colors):
            if i == 0:
                draw_text(
                    base,
                    (
                        cur_point[0] - square_size - inner_square_margin,
                        cur_point[1],
                    ),
                    "lowest",
                    position="right",
                )
            elif i == len(legend_colors) - 1:
                draw_text(
                    base,
                    (
                        cur_point[0] - square_size - inner_square_margin,
                        cur_point[1],
                    ),
                    "highest",
                    position="right",
                )
            cur_point = draw_square(base, cur_point, fill=colour)
            cur_point = (
                cur_point[0] - square_size - inner_square_margin,
                cur_point[1] + inner_square_margin + linewidth,
            )

        base.save(save_path)

        return base

    elif isinstance(model, tf.keras.Model):
        warnings.warn(
            "loading keras model.",
            Warning,
        )

        square_size = 16
        inner_square_margin = 1
        linewidth = 1
        extra_padding_bottom = 10
        max_line_squares = 16

        colors = color_palette()
        num_colors = len(colors)

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
                        cur_point = (
                            cur_point[0]
                            - (square_size + inner_square_margin + linewidth)
                            * max_line_squares,
                            cur_point[1]
                            + square_size
                            + inner_square_margin
                            + linewidth,
                        )
                    colour_index = norm_map[i] % num_colors
                    colour = colors[colour_index]
                    cur_point = draw_square(base, cur_point, fill=colour)
                    cur_point = (
                        cur_point[0] + inner_square_margin,
                        cur_point[1] - square_size - inner_square_margin,
                    )

            # add extra padding between layers
            cur_point = (
                initial_point[0],
                cur_point[1]
                + square_size
                + inner_square_margin
                + linewidth
                + extra_padding_bottom,
            )
            layer_id += 1

        base.save(save_path)

        return base
    else:
        print("Type model supported yet ")

