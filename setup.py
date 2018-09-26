# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import


#########################################################
# General config
#########################################################

# Name of the top-level package
libname="binding"

# Choose build type (optimized or debug)
build_type="optimized"

# Short description for package list on PyPI
SHORTDESC = "Equilibrium model of transcription factor chromatin-binding."

# Long description for package homepage on PyPI
DESC = open('README.md').read()

# Set up data files for packaging by defining top-level directory
datadirs  = ("test",)
dataexts  = (".py",  ".pyx", ".pxd",  ".c", ".cpp", ".h",  ".sh", ".txt")

# Standard documentation to detect (and package if it exists).
standard_docs = ["README", "LICENSE", "TODO", "CHANGELOG", "AUTHORS"]
standard_doc_exts = [".md", ".rst", ".txt", ""]


#########################################################
# Init
#########################################################

# check for Python 3.0 or later
# http://stackoverflow.com/questions/19534896/enforcing-python-version-in-setup-py
import sys
if sys.version_info < (3,0):
    sys.exit('Sorry, Python < 3.0 is not supported')

import os
import numpy as np
from setuptools import setup
from setuptools.extension import Extension

# if Cython is available, compile extensions
try:
    from Cython.Build import cythonize
    USE_CYTHON = True

# otherwise, use pre-compiled build
except ImportError:
    #sys.exit("Cython is required to build the extension modules.")
    USE_CYTHON = False
    print('Cython not found. Defaulting to pre-compiled version.')

#########################################################
# Cython options
#########################################################

if USE_CYTHON:
    ext_type = 'pyx'
    setup_requires = ["cython", "numpy"],

else:
    ext_type = 'c'
    setup_requires = ["numpy"],

#########################################################
# Compiler options
#########################################################

# Define base set of compiler and linker flags.
# This is geared toward x86_64
# See https://gcc.gnu.org/onlinedocs/gcc-4.6.4/gcc/i386-and-x86_002d64-Options.html

# Modules involving numerical computations
extra_compile_args_math_optimized    = ['-march=native', '-O2', '-msse', '-msse2', '-mfma', '-mfpmath=sse']
extra_compile_args_math_debug        = ['-march=native', '-O0', '-g']
extra_link_args_math_optimized       = []
extra_link_args_math_debug           = []

# Modules that do not involve numerical computations
extra_compile_args_nonmath_optimized = ['-O2']
extra_compile_args_nonmath_debug     = ['-O0', '-g']
extra_link_args_nonmath_optimized    = []
extra_link_args_nonmath_debug        = []

# Additional flags to compile/link with OpenMP
openmp_compile_args = ['-fopenmp']
openmp_link_args    = ['-fopenmp']


#########################################################
# Helpers
#########################################################

# Make absolute cimports work
my_include_dirs = [".", np.get_include()]

# Choose the base set of compiler and linker flags.
if build_type == 'optimized':
    my_extra_compile_args_math    = extra_compile_args_math_optimized
    my_extra_compile_args_nonmath = extra_compile_args_nonmath_optimized
    my_extra_link_args_math       = extra_link_args_math_optimized
    my_extra_link_args_nonmath    = extra_link_args_nonmath_optimized
    my_debug = False
    print( "build configuration selected: optimized" )
elif build_type == 'debug':
    my_extra_compile_args_math    = extra_compile_args_math_debug
    my_extra_compile_args_nonmath = extra_compile_args_nonmath_debug
    my_extra_link_args_math       = extra_link_args_math_debug
    my_extra_link_args_nonmath    = extra_link_args_nonmath_debug
    my_debug = True
    print( "build configuration selected: debug" )
else:
    raise ValueError("Unknown build configuration '%s'; valid: 'optimized', 'debug'" % (build_type))


def declare_extension(ext_name,
                      ext_type='pyx',
                      include_dirs=None,
                      use_math=False,
                      use_openmp=False):
    """
    Declare a Cython extension module for setuptools.

    Parameters:
        ext_name : str
            Absolute module name, e.g. use `mylibrary.mypackage.mymodule`

        ext_type : str
            Extension format, either 'pyx' or 'c'

        include_dirs : list
            Included directories.

        use_math : bool
            If True, set math flags and link with ``libm``.

        use_openmp : bool
            If True, compile and link with OpenMP.

    Return value:
        Extension object that can be passed to ``setuptools.setup``.
    """
    ext_path = ext_name.replace(".", os.path.sep)+"."+ext_type

    if use_math:
        compile_args = list(my_extra_compile_args_math) # copy
        link_args    = list(my_extra_link_args_math)
        libraries    = ["m"]  # link libm
    else:
        compile_args = list(my_extra_compile_args_nonmath)
        link_args    = list(my_extra_link_args_nonmath)
        libraries    = None  # value if no libraries

    # OpenMP args
    if use_openmp:
        compile_args.insert(0, openmp_compile_args)
        link_args.insert(0, openmp_link_args)

    # See http://docs.cython.org/src/tutorial/external.html on linking libraries to your Cython extensions.

    return Extension( ext_name,
                      [ext_path],
                      extra_compile_args=compile_args,
                      extra_link_args=link_args,
                      include_dirs=include_dirs,
                      libraries=libraries)


# Gather user-defined data files
# http://stackoverflow.com/questions/13628979/setuptools-how-to-make-package-contain-extra-data-folder-and-all-folders-inside
datafiles = []
getext = lambda filename: os.path.splitext(filename)[1]
for datadir in datadirs:
    datafiles.extend( [(root, [os.path.join(root, f) for f in files if getext(f) in dataexts]) for root, dirs, files in os.walk(datadir)] )


# Add standard documentation (README et al.), if any, to data files
detected_docs = []
for docname in standard_docs:
    for ext in standard_doc_exts:
        filename = "".join( (docname, ext) )  # relative to setup.py directory
        if os.path.isfile(filename):
            detected_docs.append(filename)
datafiles.append( ('.', detected_docs) )


# Extract __version__ from the package __init__.py
# http://stackoverflow.com/questions/2058802/how-can-i-get-the-version-defined-in-setup-py-setuptools-in-my-package
import ast
init_py_path = os.path.join(libname, '__init__.py')
version = '0.0.unknown'
try:
    with open(init_py_path) as f:
        for line in f:
            if line.startswith('__version__'):
                version = ast.parse(line).body[0].value.s
                break
        else:
            print( "WARNING: Version information not found in '%s', using placeholder '%s'" % (init_py_path, version), file=sys.stderr )
except FileNotFoundError:
    print( "WARNING: Could not find file '%s', using placeholder version information '%s'" % (init_py_path, version), file=sys.stderr )


#########################################################
# Set up extension modules
#########################################################


# declare cython extensions
ext_modules =[

    declare_extension("binding.model.elements", ext_type, my_include_dirs),

    declare_extension("binding.model.parallel", ext_type, my_include_dirs),

    declare_extension("binding.model.trees", ext_type, my_include_dirs),

    declare_extension("binding.model.partitions", ext_type, my_include_dirs)

    ]

# Call cythonize() explicitly, as recommended in the Cython documentation.
# See http://cython.readthedocs.io/en/latest/src/reference/compilation.html#compiling-with-distutils

# Note that my_ext_modules is just a list of Extension objects. We could add any C sources (not coming from Cython modules) here if needed.
if USE_CYTHON:
    ext_modules = cythonize(ext_modules,
                            include_path=my_include_dirs,
                            gdb_debug=my_debug)


#########################################################
# Call setup()
#########################################################

setup(
    name = "binding",
    version = '0.1',
    author = "Sebastian Bernasek",
    author_email = "sebastian@u.northwestern.edu",
    url = "https://github.com/sebastianbernasek/binding",
    description = SHORTDESC,
    long_description = DESC,
    license = "MIT",

    # supported platforms; http://stackoverflow.com/questions/34994130/what-platforms-argument-to-setup-in-setup-py-does
    platforms = ["Linux", "macOS"],

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers = [ "Development Status :: 4 - Beta",
                    "Environment :: Console",
                    "Intended Audience :: Science/Research",
                    "License :: MIT",
                    "Operating System :: MacOS",
                    "Programming Language :: Cython",
                    "Programming Language :: Python",
                    "Programming Language :: Python :: 3",
                    "Topic :: Scientific/Engineering",
                  ],

    # See http://setuptools.readthedocs.io/en/latest/setuptools.html
    setup_requires = setup_requires,
    install_requires = ["numpy", "scipy", "matplotlib"],
    provides = ["binding"],

    # keywords for PyPI
    keywords = ["transcription modeling regulatory syntax"],

    # Add all extension modules (list of Extension objects)
    ext_modules = ext_modules,

    # Declare packages so that  python -m setup build  will copy .py files
    packages = ["binding",
                "binding.model",
                "binding.analysis",
                "binding.utilities"],

    # Install Cython headers so that other Cython modules can cimport ours
    package_data={'binding': ['*.pxd', '*.pyx'],
                  'binding.model': ['*.pxd', '*.pyx']},

    # Disable zip_safe so cython finds .pxd within .egg
    zip_safe = False,

    # Add custom data files not inside a Python package
    data_files = datafiles
)

