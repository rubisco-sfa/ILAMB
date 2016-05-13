#!/usr/bin/env python
"""
The International Land Model Benchmarking Project (ILAMB)
"""

NAME    = 'ILAMB'
VERSION = '2.0'
AUTHOR  = 'Nathan Collier'
EMAIL   = 'nathaniel.collier@gmail.com'
DESCR   = __doc__.strip()
URL     = ''
DLURL   = URL + '/get/default.tar.gz'

def setup_package():
    from numpy.distutils.core import setup
    from numpy.distutils.core import Extension
    setup(name=NAME, version=VERSION,
          author=AUTHOR, author_email=EMAIL,
          description=DESCR, long_description=DESCR,
          url=URL, download_url=DLURL,
          packages = ['ILAMB'],
          package_dir = {'ILAMB' : 'src/ILAMB'}
          )

if __name__ == "__main__":
    setup_package()
