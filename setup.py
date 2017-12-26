# -*- coding:utf-8 -*-
from setuptools import setup, Extension
from setuptools.command.test import test as TestCommand
from Cython.Distutils import build_ext
import numpy

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        pytest.main(self.test_args)

ext_modules = [
    Extension(
        'interleaving.cProbabilistic',
        sources=[
            'interleaving/cmethods/probabilistic.pyx',
        ],
        language="c++",
    ),
    Extension(
        'interleaving.cOptimized',
        sources=[
            'interleaving/cmethods/optimized.pyx',
        ],
        language="c++",
    ),
    Extension(
        'interleaving.cRoughlyOptimized',
        sources=[
            'interleaving/cmethods/roughly_optimized.pyx',
        ],
        language="c++",
    ),
]

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
        'numpy',
        'cython'
    ],
    install_requires = [
        'numpy',
        'scipy',
        'pulp',
        'cvxopt'
    ],
    tests_require=['pytest'],
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
    cmdclass = {'test': PyTest, 'build_ext': build_ext}
)
