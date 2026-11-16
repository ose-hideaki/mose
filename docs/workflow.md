# Workflow and operations

`microevent plan CONFIG` is read-only. It validates the configuration and inputs, then prints the stage commands and declared outputs. `doctor` additionally checks whether the main optional packages can be imported; it does not prove that the native library is compiled or the sampler converges.

`microevent run CONFIG` holds a nonblocking event/version lock and executes sequentially:

1. Point models: PSPL, PSPL annual parallax, FSPL, FSPL annual parallax; each optimization is followed by local NUTS.
2. For `stage="all"`: prepare a filtered photometry copy for the external grid, execute the optional grid command, verify complete result files, collect modes, refine binary seeds.
3. Apply the binary-improvement gate. A gate that does not pass produces the point-only report rather than a pipeline failure.
4. If the gate passes: top-ranked static binary NUTS, parallax optimization/NUTS, candidate selection, optional point-model refits initialized from the selected binary, summary tables and plots.

`stage="point"` stops after point-model tables and plots. `stage="binary"` assumes the point-fit files for that version already exist. In routine use, prefer `all` or `point` in a fresh version.

Results go to `PROJECT/analysis/YEAR/EVENT/VERSION/`. Each stage has its own appended log, input/code fingerprint and output checksums in `pipeline-state.json`. `pipeline-summary.json` indicates execution completion only. Light curves have separate filenames per model; tables with the plotted points and dense curves accompany them. Caustic diagrams are produced for static and parallax binary solutions; parallax changes the source trajectory, not the static binary caustic geometry.

## External grid contract

`grid.results` names a directory of `result_*.dat` files with `SUMMARY (LM-grid)` rows. Set either `grid.expected` to the expected number of summary entries, or `grid.parameters` to the parameter-array file. This checks counts, not that the grid exhausts all physical modes. A configured `grid.command` is an argv array, never a shell command string. Supported placeholders:

- `{python}`, `{project_root}`, `{config}`
- `{grid_json}`, `{grid_list}`: generated photometry lists after the common cut
- `{results}`, `{qsub_dir}`, `{parameters}`
- `{t0}`, `{tE}`, `{u0}`: initial PSPL values

Do not point the external solver at an unfiltered original list while comparing fit statistics from a filtered pipeline. The grid adapter prepares consistent filtered files for this purpose. External async schedulers must complete before result collection; a pending/incomplete grid intentionally stops the workflow. No server-specific paths, grid maps or precompiled Linux libraries are bundled.

## Recovery and preservation

Re-running identical inputs reuses completed stages only when their output checksums match. Failed stages restart; there is no partial-chain continuation. Changing the code/configuration/data invalidates completed work. Make a new event version for a new analysis. Never mix result files from different cuts or priors. `--force` is an explicit overwrite operation for tracked outputs; back them up first.

The SGE wrapper uses the configured Python executable, queue and parallel environment. Numerical threads are limited to allocated slots; chain execution is sequential. A process can have more OS threads than allocated slots while its active numerical computation remains limited. Third-party schedulers and additional processes need their own resource controls.
