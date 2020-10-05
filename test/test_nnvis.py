from nnprint import nnprint
import torch.nn as nn
from tensorflow.keras import models

import pytest
from models import *
import shutil
import os


def test_nnvis_for_default_torch():
    """ Check if nnprint can handle torch model"""

    model = ThLeNet()
    assert isinstance(model, nn.Module)

    nnprint(model)
    assert os.path.exists("./vis01.png")
    os.remove("./vis01.png")


def test_nnvis_for_default_tf():
    """Check if nnprint can handle tf model"""

    model = TFLeNet().model()
    assert isinstance(model, models.Sequential)

    nnprint(model)
    assert os.path.exists("./vis01.png")
    os.remove("./vis01.png")


def test_nnvis_for_gm_criteria():
    """ test gm criteria for both torch and tf models"""

    model = ThLeNet()
    criteria = "gm"

    nnprint(model, importance_criteria=criteria)
    assert os.path.exists("./vis01.png")
    os.remove("./vis01.png")

    model = TFLeNet().model()
    nnprint(model, importance_criteria=criteria)
    assert os.path.exists("./vis01.png")
    os.remove("./vis01.png")


def test_nnprint_for_l2_criteria():
    """ test l2 criteria for both torch and tf models"""
    model = ThLeNet()
    criteria = "l2"

    nnprint(model, importance_criteria=criteria)
    assert os.path.exists("./vis01.png")
    os.remove("./vis01.png")

    model = TFLeNet().model()
    nnprint(model, importance_criteria=criteria)
    assert os.path.exists("./vis01.png")
    os.remove("./vis01.png")


def test_nnprint_non_default_path():
    """ test nnprint for non default path"""

    path = "./non_default.png"

    model = ThLeNet()
    nnprint(model, save_path=path)
    assert os.path.exists(path)
    os.remove(path)
