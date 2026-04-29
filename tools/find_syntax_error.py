import sys

fname = 'gui/main.py'
with open(fname, 'rb') as f:
    data = f.read()

try:
    text = data.decode('utf-8')
except Exception:
    try:
        text = data.decode('utf-8', 'replace')
    except Exception:
        text = data.decode('latin-1', 'replace')

try:
    compile(text, fname, 'exec')
    print('COMPILE_OK')
except SyntaxError as e:
    print('SyntaxError detected:')
    print('  filename:', e.filename)
    print('  lineno:', e.lineno)
    print('  offset:', e.offset)
    print('  text:', repr(e.text))
    if e.text:
        # show caret position
        print('  caret:'.ljust(8) + ' ' * (e.offset-1) + '^')
    raise
except Exception as e:
    print('Other compile error:', type(e), e)
    raise
