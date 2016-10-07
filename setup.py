# -*- coding:utf-8 -*-
from setuptools import setup
from setuptools.command.test import test as TestCommand

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        pytest.main(self.test_args)

setup(
    name = "interleaving",
    packages = ["interleaving", "interleaving.simulation"],
    version = "0.0.1",
    description = "Interleaving library for ranking evaluation",
    author = "Makoto P. Kato",
    author_email = "kato@dl.kuis.kyoto-u.ac.jp",
    license     = "MIT License",
    url = "https://github.com/mpkato/interleaving",
    entry_points='',
    install_requires = [
        'numpy',
        'scipy'
    ],
    tests_require=['pytest'],
    cmdclass = {'test': PyTest}
)
