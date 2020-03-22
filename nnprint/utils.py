import numpy as np


def color_palette(pallete="greyscale"):
    colors = pallete_greyscale
    if pallete == "purplescale":
        colors = pallete_mix

    return colors


def map_to_color(numpy_list):
    mn = np.amin(numpy_list)
    mx = np.amax(numpy_list)
    rescale = (numpy_list - mn) * (255 / (mx - mn))
    return np.around(rescale).astype(int)


pallete_purplescale = [
    "#2e3bd1",
    "#393acd",
    "#4539ca",
    "#5038c7",
    "#5c37c4",
    "#6836c1",
    "#7335be",
    "#7f34bb",
    "#8b33b8",
    "#9632b5",
    "#a231b2",
    "#ae30af",
    "#b92fac",
    "#c52ea9",
    "#d12ea6",
]

pallete_greyscale = [
    "#f0f0f0",
    "#c0c0c0",
    "#878787",
    "#474747",
    "#000000",
]
