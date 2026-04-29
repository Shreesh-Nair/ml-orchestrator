fname='gui/main.py'
with open(fname,'rb') as f:
    for i, raw in enumerate(f, start=1):
        try:
            s = raw.decode('utf-8')
        except Exception:
            s = raw.decode('latin-1')
        if s.rstrip('\r\n').endswith('\\'):
            print('Line', i, 'ends with backslash:', repr(s))
