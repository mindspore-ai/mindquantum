"""Setup the package."""
import os
from setuptools import find_packages, setup
import logging

with open("./version.txt", "r") as f:
    version = f.read().strip()


NAME = "quditop"
DESCRIPTION = "A numerically quantum simulator for qudit system."
URL = "https://github.com/forcekeng/QudiTop"
EMAIL = "forcekeng@126.com"
AUTHOR = "Li Geng, Yanzheng Zhu, Zuoheng Zou"
REQUIRES_PYTHON = ">=3.7.0"
VERSION = "0.1.0"

here = os.path.abspath(os.path.dirname(__file__))

try:
    with open(os.path.join(here, 'requirements.txt'), 'r') as f:
        required = [rq.strip() for rq in f.readlines() if rq and rq[0] != "#"]
except:
    logging.warning("Find requirements.txt failed.")
    required = []
print(here, os.path.join(here, 'requirements.txt'), required)


try:
    with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["examples", "tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=required,
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
