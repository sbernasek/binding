from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include

ext_modules = [
    Extension("microstates",["microstates.pyx"],
              include_dirs=['.']),
    Extension("partitions",["partitions.pyx"],
              include_dirs=['.'])]
              #extra_compile_args=['-fopenmp'],
              #extra_link_args=['-fopenmp'])]

setup(
    name = 'equilibrium',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    options={'build_ext':{'inplace':True, 'force':True}},
    include_dirs=[get_include()]
)

