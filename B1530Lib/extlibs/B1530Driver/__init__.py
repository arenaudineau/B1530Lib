import os, sys
sys.path.append(os.path.realpath(os.path.dirname(__file__)))

try:
	from B1530driver import *
	import B1530driver
except ModuleNotFoundError:
	raise ModuleNotFoundError("Failed to import B1530driver. Please make sure that the B1530driver files are in 'extlibs/B1530Driver'.")
else:
	# Constants export
	_operationMode = B1530driver._operationMode
	_forceVoltageRange = B1530driver._forceVoltageRange
	_measureMode = B1530driver._measureMode
	_measureVoltageRange = B1530driver._measureVoltageRange
	_measureCurrentRange = B1530driver._measureCurrentRange
	_measureEventData = B1530driver._measureEventData