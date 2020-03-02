<p align="center">
  <img src="images/logo.png">
</p>

<p align="center">
<a class="reference external" href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

<a href="https://www.python.org/downloads/">
        <img src="https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-blue"
        <img src="https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue"
            alt="Python 3.5-3.7"/></a>
<a 

[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Propaler/nnprint/master/LICENSE)
![Run Tests](https://github.com/Propaler/nnprint/workflows/Run%20tests/badge.svg?branch=master)
>
</p>

# How to install

```sh
pip install .
```

# How to use it

### For now, just one main feature is avaiable. To check this out, you can just run:

```sh
cd nnprint
python nnvis.py
```

### The defaults models is instantiated at `nnprint/models`. To change wich you want to use, just change the line 329 at `nnvis.py` to `keras`

# Output

### The output of our only visualization will probably be like this:
<p align="center">
  <img src="images/test2.png">
</p>

# Suport

### This library only suports `Torch` and `Keras` models. Check above wich layers we support from each framework:

## Torch
- Conv2d
- Linear

## Keras
- Dense

# Contribute

## Fork this repository and check our issues. A lot of `good first issues` will be probably avaiable