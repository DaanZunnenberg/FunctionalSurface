#!/usr/bin/env python
from setuptools import setup

with open('requirements.txt') as f:
    required = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]

setup(
    name='funcgarch',
    version='1.1.4',
    description='Functional GARCH and GAS-GARCH models for intraday volatility',
    author='Daan Zunnenberg',
    author_email='dw.zunnenberg@gmail.com',
    packages=['funcgarch'],
    install_requires=required,
    python_requires='>=3.10',
)
