import py_compile
import traceback
import sys

try:
    py_compile.compile('gui/main.py', doraise=True)
    print('COMPILE_OK')
except Exception:
    traceback.print_exc()
    sys.exit(1)
