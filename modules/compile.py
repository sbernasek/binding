from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include

ext_modules = [
    Extension("eq.partitions", ["eq/partitions.pyx"], include_dirs=['.']) ]

setup(
  name='equilibrium_module',
  cmdclass={'build_ext': build_ext},
  ext_modules=ext_modules,
  script_args=['build_ext'],
  options={'build_ext':{'inplace':True, 'force':True}},
  include_dirs=[get_include()]
)

print('******** CYTHON COMPILATION COMPLETE ********')

