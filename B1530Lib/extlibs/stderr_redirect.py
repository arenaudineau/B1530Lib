""" 
!! ADAPTED FROM https://gist.github.com/natedileas/8eb31dc03b76183c0211cdde57791005 !!
""" 

from contextlib import contextmanager
import io, os, sys, ctypes, tempfile

if sys.version_info < (3, 5):
	libc = ctypes.CDLL(ctypes.util.find_library('c'))
else:
	if hasattr(sys, 'gettotalrefcount'): # debug build
		libc = ctypes.CDLL('ucrtbased')
	else:
		libc = ctypes.CDLL('api-ms-win-crt-stdio-l1-1-0')

@contextmanager
def stderr_redirector(stream):
	try:
		original_stderr_fd = sys.stderr.fileno()
	except io.UnsupportedOperation: # stderr has been replaced, we fall back onto __stderr__
		original_stderr_fd = sys.__stderr__.fileno()
	
	saved_stderr_fd = os.dup(original_stderr_fd)
	try:
		tfile = tempfile.TemporaryFile(mode='w+b')

		libc.fflush(None)
		os.dup2(tfile.fileno(), original_stderr_fd)
		yield
		libc.fflush(None)
		os.dup2(saved_stderr_fd, original_stderr_fd)

		tfile.flush()
		tfile.seek(0)
		stream.write(tfile.read())
	finally:
		tfile.close()
		os.close(saved_stderr_fd)