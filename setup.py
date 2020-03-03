import setuptools

try:  # for pip >= 10
    from .pip._internal.req import parse_requirements as parse
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements as parse

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = (lambda f: [str(i.req) for i in parse(f, session=False)])(
    "requirements.txt"
)

setuptools.setup(
    name="nnprint",
    version="0.0.1",
    author="Propaler",
    author_email="jefersonnpn@gmail.com",
    description="A library for better visualization of Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Propaler/nnprint",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
)
