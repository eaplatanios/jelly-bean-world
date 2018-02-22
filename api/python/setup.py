from distutils.core import setup, Extension

simulator_c = Extension(
  'nel.simulator_c',
  define_macros = [('MAJOR_VERSION', '1'),
                   ('MINOR_VERSION', '0')],
  include_dirs = ['../..', '../../deps'],
  # libraries = ['...'],
  # library_dirs = ['/usr/local/lib'],
  sources = ['nel/simulator.cpp'], 
  extra_compile_args = ['-std=c++11'])

setup(
  name = 'nel',
  version = '1.0',
  description = 'Never-ending learning framework',
  ext_modules = [simulator_c],
  packages = ['nel'], 
  install_requires = ['enum34'])
