import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

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
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
)
