import py_compile
import glob
import sys

errs = []
for f in glob.glob('**/*.py', recursive=True):
    try:
        py_compile.compile(f, doraise=True)
    except Exception as e:
        errs.append((f, str(e)))

if errs:
    for f, e in errs:
        print('ERROR:', f, e)
    sys.exit(1)

print('OK')
