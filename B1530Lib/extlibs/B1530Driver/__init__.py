# !! Required to add to path the B1530driver files !!

try:
	from B1530driver import *
except ModuleNotFoundError:
	raise ModuleNotFoundError("Failed to import B1530driver. Please make sure that the B1530driver files are in 'extlibs/B1530Driver'.")
else:
	# Constants export
	_operationMode       = _operationMode
	_forceVoltageRange   = _forceVoltageRange
	_measureMode         = _measureMode
	_measureVoltageRange = _measureVoltageRange
	_measureCurrentRange = _measureCurrentRange
	_measureEventData    = _measureEventData