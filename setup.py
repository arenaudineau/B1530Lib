from setuptools import setup, find_packages

setup(
	name='B1530Lib',
	version='0.1.2',
	packages=find_packages(),
	install_requires=[
		'pyvisa',
		'pandas',
	]
)
