import imp
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy


__version__ = imp.load_source('multitemporal.version', 'multitemporal/version.py').__version__

extensions = [
    Extension('multitemporal.bin.*',
              ['multitemporal/bin/*.pyx'],
              include_dirs=[numpy.get_include()],)
]

setup(
    name='multitemporal',
    version=__version__,
    description='Efficient, chainable time series processing of raster stacks',
    url='https://gitlab.com/rbraswell/multitemporal.git',
    author='Bobby H. Braswell',
    author_email='rbraswell@ags.io',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Cython :: 0.26',
    ],
    license='GPLv3',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'sharedmem',
    ],
    entry_points={
        'console_scripts': [
            'multitemporal=multitemporal.mt:main'
        ]
    },
    zip_safe=False,
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(extensions),
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'numpy', 'sharedmem'],
    package_data={
        'multitemporal.test': ['*.json', '*.sh', '*.tgz']
    }
)
