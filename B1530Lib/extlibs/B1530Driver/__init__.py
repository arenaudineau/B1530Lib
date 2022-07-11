# !! Required to add B1530driver files to path !!

try:
	from B1530driver import *
	import B1530driver # for global variables
except ModuleNotFoundError:
	raise ModuleNotFoundError("Failed to import B1530driver. Please make sure to add B1530driver folder to PYTHONPATH.")
else:
	# Constants export
	_operationMode       = B1530driver._operationMode
	_forceVoltageRange   = B1530driver._forceVoltageRange
	_measureMode         = B1530driver._measureMode
	_measureVoltageRange = B1530driver._measureVoltageRange
	_measureCurrentRange = B1530driver._measureCurrentRange
	_measureEventData    = B1530driver._measureEventData
