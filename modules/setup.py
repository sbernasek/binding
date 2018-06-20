from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include
import os

#args = ['-Xpreprocessor', '-fopenmp', '-lomp']#, '-I/usr/local/opt/libomp/include', '-L/usr/local/opt/libomp/lib']
args = ['-fopenmp']

ext_modules = [
    Extension("elements",["elements.pyx"], include_dirs=['.']),

    Extension("trees",["trees.pyx"],
                include_dirs=['.'],
                extra_compile_args=args,
                extra_link_args=args),

    Extension("partitions",["partitions.pyx"],
                include_dirs=['.'],
                extra_compile_args=args,
                extra_link_args=args),

    Extension("ptest", ["ptest.pyx"],
                include_dirs=['.'],
                extra_compile_args=args,
                extra_link_args=args)
    ]

setup(
    name = 'equilibrium',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    options={'build_ext':{'inplace':True, 'force':True}},
    include_dirs=[get_include()]
)

# to compile: ARCHFLAGS='-arch x86_64' python setup.py build_ext --inplace
