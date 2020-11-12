from setuptools import setup

setup(
    name='mfs',
    version='0.1',
    description='Code for computing acoustic scattering using the method of fundamental solutions.  Computed by researchers at UC Merced.',
    # url='http://github.com/',
    author='Dustin Kleckner',
    author_email='dkleckner@ucmerced.edu',
    license='Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)',
    packages=['mfs'],
    install_requires=[ #Many of the packages are not in PyPi, so assume the user knows how to isntall them!
        'numpy',
        'scipy',
    ],
    zip_safe=False
)
