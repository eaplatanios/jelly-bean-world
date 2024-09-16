# Copyright 2019, The Jelly Bean World Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy as np

from os import environ

extra_compile_args = {
    'msvc': ['/W3', '/GT', '/Gy', '/Oi', '/Ox', '/Ot', '/Oy', '/DNDEBUG', '/DUNICODE'],
    'unix': ['-std=c++11', '-Wall', '-Wpedantic', '-Ofast', '-DNDEBUG', '-fno-stack-protector', '-mtune=native', '-march=native']}
extra_link_args = {
    'msvc': ['ws2_32.lib']}

# on Mac, if gcc is used, '-Qunused-arguments' will throw an error
if 'CFLAGS' in environ:
  environ['CFLAGS'] = environ['CFLAGS'].replace('-Qunused-arguments', '')
if 'CPPFLAGS' in environ:
  environ['CPPFLAGS'] = environ['CPPFLAGS'].replace('-Qunused-arguments', '')

class build_ext_subclass(build_ext):
  def build_extensions(self):
    c = self.compiler.compiler_type
    if c in extra_compile_args:
      for e in self.extensions:
        e.extra_compile_args = extra_compile_args[c]
    if c in extra_link_args:
      for e in self.extensions:
        e.extra_link_args = extra_link_args[c]
    build_ext.build_extensions(self)



simulator_c = Extension(
    'jbw.simulator_c',
    define_macros=[('MAJOR_VERSION', '1'), ('MINOR_VERSION', '0')],
    include_dirs=['../../jbw', '../../jbw/deps', np.get_include()],
    sources=['src/jbw/simulator.cpp'])

setup(
    name='jbw',
    version='1.0',
    license='Apache License 2.0',
    description='Jelly Bean World',
    long_description=open('../../README.md').read(),
    ext_modules=[simulator_c],
    packages=['jbw'],
    package_dir={'': 'src'},
    install_requires=['enum34'],
    cmdclass={'build_ext': build_ext_subclass})

