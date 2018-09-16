from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include
import os

import sys
for p in ('./binding/model'):
    if p not in sys.path:
        sys.path.insert(0, p)

# define extra compile arguments
args = ['-w']

# define extension modules (compilation at the binding package root directory)
ext_modules = [

    # compile binding elements submodule
    Extension("binding.model.elements",
            ["binding/model/elements.pyx"],
            include_dirs=['.'],
            extra_compile_args=args),

    # compile paralellization tools
    Extension("binding.model.parallel",
            ["binding/model/parallel.pyx"],
            include_dirs=['.'],
            extra_compile_args=args),

    # compile recursive tree submodules
    Extension("binding.model.trees",
            ["binding/model/trees.pyx"],
            include_dirs=['.', './model'],
            extra_compile_args=args,
            extra_link_args=args),

    # compile partition function submodule
    Extension("binding.model.partitions",
            ["binding/model/partitions.pyx"],
            include_dirs=['.'],
            extra_compile_args=args,
            extra_link_args=args)
    ]


setup(
    name = 'model_tools',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    script_args=['build_ext'],
    options={'build_ext':{'inplace':True, 'force':True}},
    include_dirs=[get_include()])

# to compile while supporting cython parallel threading: ARCHFLAGS='-arch x86_64' python setup.py build_ext --inplace

print('******** CYTHON COMPILATION COMPLETE ******')
