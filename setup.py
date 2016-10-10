# -*- coding:utf-8 -*-
from setuptools import setup

setup(
    name = "interleaving",
    packages = ["interleaving", "interleaving.simulation"],
    version = "0.0.1",
    description = "Interleaving library for ranking evaluation",
    author = "Makoto P. Kato, Tomohiro Manabe",
    author_email = "kato@dl.kuis.kyoto-u.ac.jp",
    license     = "MIT License",
    url = "https://github.com/mpkato/interleaving",
    setup_requires = [
        'numpy'
    ],
    install_requires = [
        'numpy',
        'scipy'
    ],
    tests_require=['pytest'],
)
