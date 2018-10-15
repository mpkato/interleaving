# -*- coding:utf-8 -*-
from setuptools import setup
from setuptools.command.test import test as TestCommand

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ["./tests"]
        self.test_suite = True

    def run_tests(self):
        import sys
        import pytest
        exit_code = pytest.main(self.test_args)
        sys.exit(exit_code)

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
        'scipy',
        'pulp'
    ],
    tests_require=['pytest'],
    cmdclass = {'test': PyTest}
)
