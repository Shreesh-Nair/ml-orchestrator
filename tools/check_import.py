import traceback
try:
    import gui.main as gm
    print('IMPORT_OK')
except Exception:
    traceback.print_exc()
    raise
