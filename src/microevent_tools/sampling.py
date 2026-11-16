"""Chain-preserving diagnostics shared by the NUTS workers."""
import numpy as np


def nuts_diagnostics(sampler):
    from numpyro.diagnostics import summary
    grouped = sampler.get_samples(group_by_chain=True)
    fields = sorted(grouped, key=lambda key: int(key.split('_')[-1]))
    raw = np.stack([np.asarray(grouped[name]) for name in fields], axis=-1)
    if raw.shape[1] >= 4:
        stats = summary(grouped, group_by_chain=True)
        rows = {name: {key: float(values[key]) if np.isfinite(values[key]) else None
                       for key in ('n_eff', 'r_hat')} for name, values in stats.items()}
    else:
        rows = {name: {'n_eff': None, 'r_hat': None} for name in fields}
    return raw, {'raw_parameter_diagnostics': rows,
        'chain_count': raw.shape[0], 'draws_per_chain': raw.shape[1],
        'scope': 'Local bounded, profiled-flux posterior. Diagnostics do not certify global mode recovery.'}
