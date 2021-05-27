from setuptools import setup
from setuptools import find_packages, find_namespace_packages
import sys 

from flexassist import __version__ 

###############################################################################
if sys.version_info < (3, 7):
	print('{} requires Python 3.7 or later.'.format('flexassist-core'))

###############################################################################

# list required packaged
REQUIRED_PACKAGES = [
]

# read description for long-form description
with open('README.md') as f:
    readme = f.read()


### development mode, symlinks and editable
# python setup.py develop
# pip    install -e .
setup(
    name='flexassist-core',
    version=__version__,            
    description='Encapsulation of methods for AI Best practices',
    keywords='deep learning, mlops',

    license='MIT',
    author='Ari Kamlani',
    author_email='akamlani@gmail.com',
    classifiers=['Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
    url='https://github.com/akamlani/flexassist',

    # what is packaged here
    install_requires=REQUIRED_PACKAGES,
    packages=find_namespace_packages(include=["flexassist.*"]),
   	# What to include
   	package_data={
        '': ['*.txt', '*.rst', '*.md']
    },
    # Testing
    test_suite='tests',
   	tests_require=[
        'pytest',
    ],
)


