#!/usr/bin/env python3

from setuptools import setup, find_packages
from pathlib import Path

classifiers = """
Development Status :: 3 - Alpha
Intended Audience :: Developers
Operating System :: OS Independent
Programming Language :: Python :: 3
Topic :: Software Development :: Libraries :: Python Modules
Topic :: Utilities
"""

__doc__ = """A preprocessing library for TMO.

The code repository is located at <https://github.com/lcls-users/tmo-prefex>
"""

requires = list(filter(
                lambda x: x and not '+' in x,
                (Path(__file__).parent/"requirements.txt")
                    .read_text()
                    .split('\n')
               ))

setup(
    name='tmo-prefex',
    version='0.1.0',
    description=__doc__.split('\n', 1)[0],
    long_description = __doc__,
    author=['Ryan Coffee', 'David M. Rogers'],
    author_email=['moc.liamg@hcemtatsevitciderp'[::-1]],
    keywords='detector data',
    url = 'http://github.com/lcls-users/tmo-prefex',
    install_requires = requires,
    entry_points={
        # make the scripts available as command line scripts
        "console_scripts": [
            "fex2h5 = tmo_prefex.cmd.fex2h5:run",
            "concat_prefex = tmo_prefex.cmd.concat_prefex:run",
        ]
    },
    #packages=find_packages(),       # Automatically find packages in the directory
    packages=['tmo_prefex'],             # List of packages to include
    classifiers=list(filter(None, classifiers.split("\n"))),
    platforms=['any'],
    python_requires='>=3.9',
)
