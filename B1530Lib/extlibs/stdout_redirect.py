"""
!! ADAPTED FROM https://gist.github.com/natedileas/8eb31dc03b76183c0211cdde57791005 !!
"""

from contextlib import contextmanager
import io, os, sys, ctypes, tempfile

if sys.version_info < (3, 5):
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
else:
    if hasattr(sys, "gettotalrefcount"):  # debug build
        libc = ctypes.CDLL("ucrtbased")
    else:
        libc = ctypes.CDLL("api-ms-win-crt-stdio-l1-1-0")


@contextmanager
def stdout_redirector(stream):
    try:
        original_fd = sys.stdout.fileno()
    except (
        io.UnsupportedOperation
    ):  # stdout has been replaced, we fall back onto __stdout__
        original_fd = sys.__stdout__.fileno()

    saved_fd = os.dup(original_fd)
    try:
        tfile = tempfile.TemporaryFile(mode="w+b")

        libc.fflush(None)
        os.dup2(tfile.fileno(), original_fd)
        yield
        libc.fflush(None)
        os.dup2(saved_fd, original_fd)

        tfile.flush()
        tfile.seek(0)
        stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_fd)
