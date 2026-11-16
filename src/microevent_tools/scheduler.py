"""Generate and optionally submit one Grid Engine coordinator job."""
import os
import shlex
import subprocess
from .execution import environment
from .paths import EventPaths


def write_script(config, output):
    output.parent.mkdir(parents=True,exist_ok=True)
    env=environment(config)
    lines=['#!/usr/bin/env bash','set -euo pipefail',f'cd {shlex.quote(str(config.project_root))}']
    for key in ('PYTHONPATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','PYTHONUNBUFFERED'):
        lines.append(f'export {key}={shlex.quote(env[key])}')
    lines.append(shlex.join(['nice','-n',str(config.nice),config.python,'-m','microevent_tools','run',str(config.config_file)]))
    output.write_text('\n'.join(lines)+'\n');output.chmod(0o755)
    return output


def submit(config):
    paths=EventPaths.from_settings(config);folder=paths.output/'scheduler'
    script=write_script(config,folder/'pipeline.sh')
    command=[config.qsub,'-cwd','-S','/bin/bash','-pe',config.parallel_environment,str(config.slots),
             '-N',f'{config.event}_mcmc','-j','y','-o',str(folder/'pipeline.log')]
    if config.queue:command+=['-q',config.queue]
    result=subprocess.run([*command,str(script)],cwd=config.project_root,check=True,capture_output=True,text=True)
    print(result.stdout,end='')
    return script
