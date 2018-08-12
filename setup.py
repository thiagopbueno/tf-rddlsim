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
    license='GNU General Public License v3.0',
    keywords=['rddl', 'tensorflow', 'probabilistic-planning', 'mdp', 'simulator'],
    url='https://github.com/thiagopbueno/tf-rddlsim',
    packages=find_packages(),
    scripts=['scripts/tfrddlsim'],
    install_requires=[
        'pyrddl',
        'matplotlib',
        'numpy',
        'tensorflow',
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
