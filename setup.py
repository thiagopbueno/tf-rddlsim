# This file is part of tf-rddlsim.

# tf-rddlsim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# tf-rddlsim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with tf-rddlsim. If not, see <http://www.gnu.org/licenses/>.

import tfrddlsim

import os
from setuptools import setup, find_packages


def read(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    file = open(filepath, 'r')
    return file.read()


setup(
    name='tfrddlsim',
    version=tfrddlsim.__version__,
    author='Thiago P. Bueno',
    author_email='thiago.pbueno@gmail.com',
    description='RDDL2TensorFlow parser, compiler, and simulator.',
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    license='GNU General Public License v3.0',
    keywords=['rddl', 'tensorflow', 'probabilistic-planning', 'mdp', 'simulator'],
    url='https://github.com/thiagopbueno/tf-rddlsim',
    packages=find_packages(),
    scripts=['scripts/tfrddlsim', 'tfrddlsim/viz/plots/plot_navigation_policy'],
    install_requires=[
        'matplotlib',
        'numpy',
        'pyrddl>=0.1.8',
        'rddlgym>=0.5.8',
        'rddl2tf>=0.5.1',
        'tensorflow<=1.15',
        'tensorflow-tensorboard',
        'typing'
    ],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
)
