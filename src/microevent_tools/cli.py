"""One entry point for planning, execution, cluster submission and native fits."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from .config import load


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['native-fit']:
        from .kernels.native_fit import main as native
        sys.argv = ['microevent native-fit', *argv[1:]]
        native()
        return 0
    p=argparse.ArgumentParser(prog='microevent',description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('plan','run','submit','status','doctor','_grid'):
        item=sub.add_parser(name)
        item.add_argument('config',type=Path)
        if name=='run':item.add_argument('--force',action='store_true')
    item=sub.add_parser('native-fit',help='Standalone VBMicrolensing + emcee fitter')
    item.add_argument('args',nargs=argparse.REMAINDER)
    args=p.parse_args(argv)
    try:
        if args.command=='native-fit':
            from .kernels.native_fit import main as native
            sys.argv=['microevent native-fit',*args.args];native();return 0
        config=load(args.config)
        if args.command=='plan':
            from .pipeline import describe
            print(json.dumps(describe(config),indent=2))
        elif args.command=='run':
            from .pipeline import execute
            print(json.dumps(execute(config,force=args.force),indent=2))
        elif args.command=='submit':
            from .scheduler import submit
            submit(config)
        elif args.command=='_grid':
            from .grid import prepare_and_execute
            prepare_and_execute(config)
        elif args.command=='status':
            from .paths import EventPaths
            path=EventPaths.from_settings(config).output/'pipeline-state.json'
            print(path.read_text() if path.exists() else '{"status":"not_started"}')
        elif args.command=='doctor':
            from .pipeline import describe
            result=describe(config)
            result['dependencies']={name:importlib.util.find_spec(name) is not None for name in ('jax','numpyro','lcbinint')}
            print(json.dumps(result,indent=2))
            return 0 if all(result['dependencies'].values()) else 2
        return 0
    except (ValueError,OSError,RuntimeError,ModuleNotFoundError) as error:
        print(f'ERROR: {error}',file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
