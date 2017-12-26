# -*- coding:utf-8 -*-
from setuptools import setup, Extension
from setuptools.command.test import test as TestCommand

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        pytest.main(self.test_args)

include_dirs = []
cmdclass = {'test': PyTest }
ext_modules = []
try:
    import numpy
    from Cython.Distutils import build_ext
    include_dirs.append(numpy.get_include())
    cmdclass['build_ext'] = build_ext
    ext_modules.append(Extension('interleaving.cProbabilistic',
            sources=['interleaving/cmethods/probabilistic.pyx'],
            language="c++"))
    ext_modules.append(Extension('interleaving.cOptimized',
            sources=['interleaving/cmethods/optimized.pyx'],
            language="c++"))
    ext_modules.append(Extension('interleaving.cRoughlyOptimized',
            sources=['interleaving/cmethods/roughly_optimized.pyx'],
            language="c++"))
except ImportError:
    pass

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
    ],
    install_requires = [
        'numpy',
        'scipy',
        'pulp',
        'cvxopt'
    ],
    tests_require=['pytest'],
    ext_modules=ext_modules,
    include_dirs=include_dirs,
    cmdclass=cmdclass
)
