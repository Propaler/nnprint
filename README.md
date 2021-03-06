<p align="center">
  <img src="https://raw.githubusercontent.com/Propaler/nnprint/master/images/logo.png">
</p>

<p align="center">
<a class="reference external" href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

<a href="https://www.python.org/downloads/">
        <img src="https://img.shields.io/badge/python-3.6%20%7C%203.7-blue"
             alt="Python 3.6-3.7"/></a>
<a 

[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Propaler/nnprint/master/LICENSE)
![Run Tests](https://github.com/Propaler/nnprint/workflows/Run%20tests/badge.svg?branch=master)
>
</p>

---

## How to install

First clone this repo.

```sh
cd nnprint
pip install .
```

## How to use it

For now, just one main feature is avaiable. To check this out, you can just run:

```sh
from nnprint import nnprint

model = <torch, tensorflow>

nnprint(model)
```

**nnprint Arguments**

- `model` - model that will be printed
- `importance_criteria` - way that weights will be classified
- `save_path` - file that the image will be save
- `title_font_size` - title font size
- `title` - graph title
- `subtile` - graph subtitle



The defaults models is instantiated at `nnprint/models`. To change wich you want to use, just change the line 329 at `nnvis.py` to `keras`

## Output

The output of our only visualization will probably be like this:
<p align="center">
  <img src="https://raw.githubusercontent.com/Propaler/nnprint/master/images/lenet_torch.png">
</p>

## Suport

 This library only suports `Torch` and `Keras` models. Check above wich layers we support from each framework:

### Torch
- Conv2d
- Linear

### Keras
- Dense

## Contribute

Fork this repository and check our issues. A lot of `good first issues` will be probably avaiable
