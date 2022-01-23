#!/usr/bin/env python3

from setuptools import setup, Command, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='Deep_Nav',
      version='1.0.0',
      description='Deep Learning Agents',
      license='As Is',
      author='Julian Bolivar',
      author_email='bolivartech.com@gmail.com',
      url='https://github.com/Unity-Technologies/ml-agents',
      packages=find_packages(),
      install_requires = required,
      long_description= ("Navigation Agent based on Deep Learning")
     )
