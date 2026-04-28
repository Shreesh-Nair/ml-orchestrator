import importlib, pathlib, sys
pkgs=['numpy','scipy','sklearn','pandas','numexpr']
for p in pkgs:
    try:
        m=importlib.import_module(p)
        v=getattr(m,'__version__',None)
        ppth=pathlib.Path(m.__file__).parent
        exts=list(ppth.glob('**/*.pyd'))+list(ppth.glob('**/*.dll'))+list(ppth.glob('**/*.so'))
        print(f"{p}: version={v}, path={ppth}, ext_count={len(exts)}")
        for e in exts[:20]:
            print('  ', e)
    except Exception as e:
        print(f"{p}: not installed or import error: {e}")
