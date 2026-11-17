# mose

mose is a Python toolkit for reproducible gravitational microlensing analysis. It combines model fitting, local MCMC sampling, and diagnostic plots in a configuration-driven workflow that runs locally or through Grid Engine.

Binary-lens magnifications and gradients are evaluated with the differentiable CPU backend of [lcbinint](https://github.com/NunotaKansuke/lcbinint). Single-lens models use an analytic point-source formula and uniform-source image-contour integration implemented in JAX.

This is an early developer release. The command-line entry point is **`microevent`**.

## Features

- Point-source point-lens (PSPL) and finite-source point-lens (FSPL) fitting
- Static and annual-parallax models
- Binary-lens multistart refinement from external grid-search results
- Local posterior sampling with NumPyro NUTS
- Model-comparison tables, light curves, residuals, and caustic diagrams
- Versioned outputs, per-stage logs, and reuse of unchanged completed stages
- Optional Grid Engine submission with configurable CPU resources

The binary workflow is:

```text
Point-lens fits → external grid search → binary refinement
    → improvement check → local static/parallax NUTS → reports and plots
```

## Developer Install

Requirements:

- Python 3.11 or later and Git
- A C++17 compiler and CMake 3.16 or later
- GSL development headers/libraries and Boost headers
- CPU JAX/JAXlib 0.6.2 and NumPyro 0.19.0

Install the native development dependencies first:

```bash
# macOS
brew install gsl boost
```

```bash
# Debian / Ubuntu
sudo apt-get install libgsl-dev libboost-dev
```

Then install mose in a separate environment:

```bash
git clone https://github.com/ose-hideaki/mose.git
cd mose

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install 'numpy<2' 'jax==0.6.2' 'numpyro==0.19.0' scikit-build-core pybind11 cmake
CMAKE_BUILD_PARALLEL_LEVEL=2 python -m pip install --no-build-isolation -e '.[jax,test]'
```

The installation builds a pinned lcbinint source revision. No GPU is required. Installing JAX before building without isolation keeps the FFI headers aligned with the runtime.

If Homebrew libraries are not found on macOS, replace the last command with:

```bash
GSL_ROOT="$(brew --prefix gsl)" CMAKE_PREFIX_PATH="$(brew --prefix)" \
  CMAKE_BUILD_PARALLEL_LEVEL=2 python -m pip install --no-build-isolation -e '.[jax,test]'
```

See [Backend notes](docs/backend-lcbinint.md) for the pinned revision and numerical details.

## Quick Use

Run the bundled synthetic example from the repository root:

```bash
microevent plan examples/point.toml
microevent doctor examples/point.toml
microevent run examples/point.toml
microevent status examples/point.toml
```

- `plan` validates the inputs and prints the workflow without running it.
- `doctor` also checks the main backend dependencies.
- `run` fits and samples the four single-lens models: PSPL, FSPL, and their annual-parallax counterparts.
- `status` displays the saved stage records.

The example uses two synthetic flux datasets. Its sampling settings are 500 warmup steps and 1,000 retained draws per chain, with two sequential chains. Results are written to:

```text
examples/demo/analysis/2026/DEMO/v1/
├── logs/                   # One log per stage
├── result/                 # Best fits, chains, tables, and plots
├── pipeline-state.json     # Stage records and checksums
└── pipeline-summary.json   # Execution summary
```

Repeating `run` reuses completed stages when their inputs, code, and outputs match. An interrupted sampling stage restarts; it does not extend a partial chain. Use a new event version when changing the data or analysis settings.

## Analyze Your Own Event

Create a project with this layout:

```text
project/
├── event.toml
└── data/2026/EVENT/
    ├── RADec
    └── v1/
        ├── list
        ├── EVENT-0
        └── photometry.flux
```

A minimal `event.toml` is:

```toml
[event]
name = "EVENT"
year = "2026"
version = "v1"
project_root = "."

[pipeline]
stage = "point"

[mcmc]
warmup = 500
samples = 1000
chains = 2
seed = 42
```

Paths are resolved relative to the configuration file. Each row of `list` specifies a photometry file, observatory, and band:

```text
photometry.flux OBSERVATORY I
```

Photometry files contain `time flux sigma` in their first three columns. Use days after JD/HJD 2450000 consistently; the loader does not convert time standards. `RADec` contains, for example:

```text
RA= 18:00:00 Dec= -29:00:00
```

`EVENT-0` supplies initial parameters in the legacy `#BEST` format. Copy the structure of [the synthetic initial model](examples/demo/data/2026/DEMO/v1/DEMO-0) and replace its values. Then run:

```bash
microevent plan project/event.toml
microevent run project/event.toml
```

For binary analysis, start from [examples/full.toml](examples/full.toml). Supply compatible external autogrid results or a grid command, and configure the expected result count. The grid solver and magnification maps are not bundled. See [Workflow details](docs/workflow.md).

## Grid Engine

Configure the queue and parallel environment in `[scheduler]`, then submit:

```bash
microevent submit project/event.toml
```

The defaults are two CPU slots and nice level 19; numerical threads are limited to the allocated slots. Only `submit` calls `qsub`. External grid-search jobs need their own resource settings.

## Tests and Documentation

```bash
python -m unittest discover -s tests -v
```

Optional tests are skipped when their dependencies are unavailable. The separate legacy `native-fit` comparison tool requires `pip install -e '.[native]'`; it uses VBMicrolensing and emcee, which are not required by the default pipeline.

- [Scientific conventions and inference scope](docs/scientific-scope.md)
- [Workflow, external grid integration, and recovery](docs/workflow.md)
- [lcbinint backend](docs/backend-lcbinint.md)
- [Validation record](docs/validation.md)
- [Migration notes and source attribution](docs/migration.md)

Local NUTS uses bounded priors around fitted solutions and profiled source/blend fluxes. Execution completion does not establish convergence or recovery of every degenerate mode. Alpha is in radians; new annual-parallax files use north/east components and declare `# parallax_basis= NE`. Old VBM-basis files require conversion or a fresh fit.

The primary project does not currently declare a redistribution license. External dependencies retain their respective licenses.
