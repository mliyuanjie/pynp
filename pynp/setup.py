from glob import glob
from setuptools import find_packages, setup 


setup (
    name = "pynp",
    version="1.0.0",
    description="Python package for nanopore data analysis",
    author="yuanjie li",
    author_email="liyuanjie777@gmail.com",
    install_requires=['numpy', 'matplotlib', 'h5py', 'statsmodels', 'scipy', 'pandas', 'lmfit', 'nidaqmx'],
    packages=['pynp', 'pynp.app', 'pynp.daq'],
	package_data = {'pynp': ['cfunction.pyd']}
)