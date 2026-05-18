# EE488 Final — Inverse Design of a Polarization-Basis Unitary Operator

A Tidy3D inverse-design project that compiles an arbitrary 2×2 unitary
operator (default: Hadamard) into a ~12 µm × 6 µm patterned-silicon device.
Light enters one end of a 220 nm SOI waveguide as a superposition of TE₀ and
TM₀ modes, propagates through the inverse-designed region, and exits
transformed by the target unitary.

See [`NOTEBOOK_EXPLANATION.md`](NOTEBOOK_EXPLANATION.md) for a pedagogical
walkthrough of the physics, the optimization loop, and the verification
strategy.

---

## Repository layout

```
EE488_Final/
├── README.md                      # this file
├── NOTEBOOK_EXPLANATION.md        # physics + algorithm walkthrough
├── environment.yml                # conda/mamba environment specification
├── polarization_unitary.ipynb     # the project (Jupyter notebook)
├── misc/                          # checkpoints, GDS exports (generated)
└── data/                          # FDTD .hdf5 results (generated)
```

## Prerequisites

1. **mamba** (recommended) or **conda**. Install
   [Miniforge](https://github.com/conda-forge/miniforge) to get mamba.
2. A **Flexcompute Tidy3D account**. Tidy3D simulations run on Flexcompute's
   cloud and consume FlexCredits. Sign up at
   [tidy3d.simulation.cloud](https://tidy3d.simulation.cloud) and grab your
   API key from your account page.

## Installation

```bash
git clone <repo-url> EE488_Final
cd EE488_Final

mamba env create -f environment.yml
mamba activate tidy3d_env
```

This installs Python 3.10, Tidy3D 2.10, autograd, optax, jax, and the
matplotlib/numpy/scipy stack with versions pinned to those that produced
the original results.

### Configuring the Tidy3D API key

One-time setup so `tidy3d.web` can submit jobs to the cloud:

```bash
tidy3d configure                # interactive prompt — paste your API key
```

Verify it works:

```bash
python -c "import tidy3d.web as web; print(web.test())"
```

You should see a "successfully tested API key" message.

## Running the project

The project lives in `polarization_unitary.ipynb`. Launch Jupyter and open it:

```bash
jupyter lab polarization_unitary.ipynb
```

Run cells top to bottom. The optimization loop (Section 15) checkpoints
itself to `misc/polarization_unitary_history.pkl` every iteration, so you
can interrupt and resume freely — re-running the loop cell picks up where
it left off.

Expect a few hours wall time for the full 60-iteration run and ~60–120
FlexCredits on the Flexcompute cloud. Outputs land in `misc/` (checkpoint +
GDS) and `data/` (`.hdf5` simulation results).

## What to expect

- Section 11 prints an upfront cost estimate (`web.estimate_cost`). Check it
  before kicking off the loop.
- Section 15 prints `iteration / total, beta, J, |grad|` per step. J ranges
  in [0, 4]; values near 4 indicate the device is approaching the Hadamard.
- Section 17 prints the extracted 2×2 unitary, the elementwise error
  against the target, the fidelity F, transmission T, and the unitarity
  residual.
- Section 18 runs six canonical input states through the final device and
  reports per-state fidelity.
- Section 20 exports the binarized layout to `misc/polarization_unitary.gds`.

## Changing the target unitary

`polarization_unitary.ipynb` Section 12 defines `U_target = hadamard()`.
Helpers are provided for `pauli_x()`, `pauli_z()`, `rotator(alpha)`, and
`retarder(phi)`. To design for a different unitary, swap that one line
(and delete `misc/polarization_unitary_history.pkl` so the optimizer
starts fresh):

```python
U_target = pauli_x()                         # TE <-> TM swap
# or
U_target = rotator(np.pi/4) @ retarder(np.pi/2) @ rotator(np.pi/4)
```

## Updating the environment

If you add a new dependency, regenerate `environment.yml`:

```bash
mamba env export -n tidy3d_env --from-history > environment.yml
```

Collaborators can sync with:

```bash
mamba env update -f environment.yml --prune
```

## Troubleshooting

- **`ModeSolver` returns the wrong polarization order** — Section 6
  empirically detects which `mode_index` is TE vs TM by inspecting the
  modal `|Ey|²` / `|Ez|²` fractions. If the printed assignments look
  wrong, plot the mode profiles in Section 6 and verify by eye.
- **TM mode is poorly confined** — the default cross-section (220 nm × 500 nm)
  bounds TM₀ but weakly. If you see leakage, widen the waveguide
  (`w_width = 0.6` µm) in Section 2.
- **Optimization plateaus at the diagonal solution** — the device may not
  have enough z-asymmetry to couple TE↔TM. The default setup has air above
  and SiO₂ below the slab, which provides the asymmetry. If you change the
  cladding to symmetric, expect coupling to vanish.
- **`mamba env create` fails on Tidy3D** — try installing Tidy3D
  separately after env creation: `pip install tidy3d==2.10.1`.
