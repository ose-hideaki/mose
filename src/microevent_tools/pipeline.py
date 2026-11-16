"""Plan and execute a versioned microlensing workflow without cluster assumptions."""
from pathlib import Path
import json
import shlex
from .data import load_datasets, read_best
from .paths import EventPaths, suffix
from .execution import Step, Runner, kernel
from .stages import point, binary, report


def plan(config):
    paths = EventPaths.from_settings(config)
    for required in (paths.initial, paths.data_list, paths.radec):
        if not required.is_file():
            raise FileNotFoundError(required)
    center = read_best(paths.initial)['t0']
    datasets = load_datasets(paths.data_list, center, config.time_window)
    count = sum(len(d.values) for d in datasets)
    # Legacy workers use a finite half-window; explicitly include all valid epochs by default.
    if config.time_window is None:
        import numpy as np
        window = float(np.nextafter(max(float(np.max(np.abs(d.values[:,0]-center))) for d in datasets), np.inf))
    else:
        window = config.time_window
    steps = []
    if config.stage in ('point', 'all'):
        steps += point.build(config, paths, center, window)
    if config.stage in ('binary', 'all'):
        # This command must finish its grid before mode collection. Partial results fail explicitly.
        steps.append(Step('external_grid', (config.python, '-m', 'microevent_tools', '_grid', str(config.config_file)),
            (), (paths.point('PSPL'),)))
        steps.append(Step('collect_grid_modes', kernel(config, 'collect_autogrid_modes',
            '--result-dir', config.grid_results, '--template', paths.point('PSPL'),
            '--output-prefix', paths.modes, '--ndata', count, '--max-modes', config.max_modes),
            (suffix(paths.modes,'.candidates'), suffix(paths.modes,'.mode1')),
            (paths.point('PSPL'),)))
        steps += binary.build(config, paths, center, window)
        if config.post_binary:
            steps += point.build(config, paths, center, window, post=True)
        steps += report.build(config, paths, center, window, point_only=True)
        steps += report.build(config, paths, center, window, point_only=False, post=config.post_binary)
        for label, model in [('STATIC', suffix(paths.binary,'.rank1.MCMC.bestfit')),
                             ('PARALLAX', suffix(paths.binary,'.rank1.PARALLAX.VBM.MCMC.bestfit'))]:
            image = paths.output/'result'/f'{paths.model}.{label}_CAUSTIC.png'
            steps.append(Step(f'caustic_{label.lower()}', kernel(config,'plot_caustic',
                '--model',model,'--output',image,
                *(('--parallax', '--radec', paths.radec) if label=='PARALLAX' else ())), (image,), (model,), 'binary'))
    else:
        steps += report.build(config, paths, center, window, point_only=True)
    return paths, datasets, steps


def describe(config):
    paths, datasets, steps = plan(config)
    return dict(event=config.event, output=str(paths.output), data_points=sum(len(d.values) for d in datasets),
        slots=config.slots, nice=config.nice,
        steps=[dict(name=s.name,condition=s.condition,command=shlex.join(s.command),
                    outputs=list(map(str,s.outputs))) for s in steps])


def execute(config, *, force=False):
    import fcntl
    paths, datasets, steps = plan(config)
    paths.output.mkdir(parents=True, exist_ok=True)
    with (paths.output/'.pipeline.lock').open('w') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('This event/version is already running') from error
        runner=Runner(config,paths.output,[paths.data_list,paths.initial,paths.radec,*[d.path for d in datasets]],force=force)
        for step in steps:
            if step.name == 'binary_refine':
                candidates = suffix(paths.modes, '.candidates')
                values = tuple(Path(line.strip()) for line in candidates.read_text().splitlines() if line.strip())
                step=Step(step.name,step.command,step.outputs,step.inputs+values,step.condition,step.accepted_codes)
            # Include completed external grid contents in mode collection provenance.
            if step.name in ('collect_grid_modes','external_grid'):
                from .grid import verify_results
                if step.name=='collect_grid_modes':
                    files=verify_results(config)
                    step=Step(step.name,step.command,step.outputs,step.inputs+tuple(files),step.condition,step.accepted_codes)
                elif config.grid_results and config.grid_results.is_dir():
                    # Re-verify completeness even if external submission was cached.
                    old=runner.state['steps'].get(step.name)
                    if old and old.get('status')=='complete':verify_results(config)
            runner.run(step)
        result=dict(status='pipeline_complete',event=config.event,output=str(paths.output),
                    note='Execution completion is not a convergence or global-mode-recovery guarantee.')
        (paths.output/'pipeline-summary.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
