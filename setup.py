#!/usr/bin/env python
"""Setup script for the RTBM package
"""

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy
import os

extensions = [
    Extension('rtbm.riemann_theta.riemann_theta',
              sources=[os.path.join('rtbm','riemann_theta','riemann_theta.pyx'),
                       os.path.join('rtbm','riemann_theta','finite_sum.c')],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-std=c99'],
    ),    
    Extension('rtbm.riemann_theta.radius',
              sources=[os.path.join('rtbm','riemann_theta','radius.pyx'),
                       os.path.join('rtbm','riemann_theta','lll_reduce.c')],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-std=c99'],
    ),
    Extension('rtbm.riemann_theta.integer_points',
              sources=[os.path.join('rtbm','riemann_theta','integer_points.pyx')],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-std=c99'],
    ),
]

setup(
    name='RTBM',
    description='Riemann-Theta Boltzmann Machine',
    version='0.1',
    author='S. Carrazza, D. Krefl',
    author_email='stefano.carrazza@cern.ch',
    packages=['rtbm'],
    ext_modules=cythonize(extensions),
    url='https://github.com/scarrazza/RTBM',
    license='LICENSE.md',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.13",
        "cma >= 2.0.0",
    ],
    zip_safe=False,
)
