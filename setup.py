from setuptools import setup, find_packages

setup(
    name="supicker",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "tifffile>=2023.1.1",
        "numpy>=1.24.0",
        "tensorboard>=2.12.0",
    ],
    python_requires=">=3.10",
)
