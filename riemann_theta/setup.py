from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

s0 = ['riemann_theta.pyx','finite_sum.c']
s1 = ['radius.pyx','lll_reduce.c']
s2 = ['integer_points.pyx']

extensions = [
    Extension("riemann_theta", s0,include_dirs=[numpy.get_include()]),
    Extension("radius", s1,include_dirs=[numpy.get_include()]),
    Extension("integer_points", s2,include_dirs=[numpy.get_include()])
]


setup(
  name = 'RTBM RiemannTheta',
  ext_modules = cythonize(extensions),
)
