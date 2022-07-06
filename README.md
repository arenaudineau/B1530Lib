# B1530Lib
Python library to easily control the Keysight B1530A, developped @ C2N

## Installation
This library requires:  
* ⚠️ 32 bits version of Python
* `pyvisa` version 1.12.0 or any other compatible version
* `pandas` version 1.4.2 or any other compatible version
* The associated NI-VISA drivers, see [the official doc](https://pyvisa.readthedocs.io/en/latest/faq/getting_nivisa.html#faq-getting-nivisa). (⚠️ 32 bits version required)

### Global installation
1. Download and install the NI-VISA drivers
2. `B1530driver.py` and `B1530ErrorModule.py` are licensed and cannot be shared on GitHub, they are therefore missing on this repo.  
You must add their path to the environment variable PYTHONPATH. See the end of this README for instructions.  
3. Run the command `pip install https://github.com/arenaudineau/B1530Lib/archive/refs/heads/main.zip`

You can now use `B1530Lib` as a regular library, by using `import B1530Lib` in any directory on the computer.

### Extending the driver
You need to create a fork of this repo, `git clone` your fork onto your local computer and run `pip install -e .` in the root of the downloaded folder.  
You can know use `B1530Lib` in any directory of the computer and any changes in the sources will be taken into account. 

### Adding path to PYTHONPATH
TODO

# Wiki
Here is a complete [wiki](../../wiki) on how to use this library.
