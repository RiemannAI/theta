from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

s0 = ['riemann_theta.pyx','finite_sum.c']
s1 = ['radius.pyx','lll_reduce.c']
s2 = ['integer_points.pyx']

extensions = [
    Extension("riemann_theta", s0),
    Extension("radius", s1),
    Extension("integer_points", s2)
]


setup(
  name = 'RTBM RiemannTheta',
  ext_modules = cythonize(extensions),
)
