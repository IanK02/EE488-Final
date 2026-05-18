# What this notebook builds — a pedagogical tour

## 1. The big picture: a "polarization gate" on a chip

In quantum information, a **qubit** is any two-level system, and a **single-qubit gate** is a 2×2 unitary matrix that rotates the state vector. The classic optical realization uses bulk waveplates: half-wave plates rotate polarization, quarter-wave plates add phase between polarizations, and stacking them produces any U ∈ U(2).

This notebook builds the *integrated-photonic* equivalent — a single ~12 µm × 6 µm patch of silicon that, when light enters one end of a waveguide and exits the other, has its polarization state transformed by a chosen 2×2 unitary. The **two basis states are the TE₀ and TM₀ guided modes** of the waveguide. Any optical state in this two-dimensional Hilbert space can be written

$$|\psi\rangle = a_{\rm TE}\,|{\rm TE}_0\rangle + a_{\rm TM}\,|{\rm TM}_0\rangle$$

and the device's job is to take the input vector `(a_TE, a_TM)` and produce the output vector `U_target · (a_TE, a_TM)`, where here `U_target` is the Hadamard.

**Why Hadamard?** Because it's the "create a superposition" gate — it maps a pure TE input to an equal superposition of TE and TM with a specific relative phase, and vice versa. It's nontrivial: it requires both **mode mixing** (off-diagonal coupling) and a **relative phase** (the −1 in the bottom-right). A device that can do Hadamard can, by composition, do any single-qubit gate.

## 2. The geometry: where the device lives

```
            ┌────────────────────────────┐
   ─────────┤                            ├─────────
    TE/TM  →│   Inverse-designed Si     │  → TE/TM
   ─────────┤   blob (12 µm × 6 µm)     ├─────────
            └────────────────────────────┘
                  220 nm Si on SiO₂
```

- **Input waveguide** (left): a 220 nm × 500 nm strip of Si on a 2 µm SiO₂ BOX. This cross-section supports **both** TE₀ (Ey-dominant) and TM₀ (Ez-dominant). Light enters carrying any superposition of these two modes.
- **Design region** (middle): a 12 µm × 6 µm patch where the 220 nm Si layer is allowed to be patterned arbitrarily — every 30 nm pixel can be either Si or air. This is the "blob" the optimizer designs.
- **Output waveguide** (right): identical to the input.

The whole device is just one slab thickness — *patterning happens only in the xy plane*. The vertical (z) asymmetry (air above, SiO₂ below) is what lets the device couple TE↔TM in the first place: a perfectly vertically symmetric structure would have TE and TM as exact eigenmodes that can never mix.

## 3. How the device actually does its job

Forget inverse design for a moment. Think about what physics makes polarization rotation possible at all in a slab waveguide:

1. **Differential propagation phase (the "retarder" effect).** TE₀ and TM₀ have different effective indices `n_TE ≈ 2.4`, `n_TM ≈ 1.7`. After propagating a distance L, TM accumulates an extra phase `(n_TE − n_TM) · 2π L / λ` relative to TE. So *any* straight waveguide section is a phase retarder.
2. **Mode mixing (the "rotator" effect).** When Si is removed from some places and not others, the resulting bumpy boundary scatters the field. Because the boundary breaks the original symmetry, a wave coming in as pure TE can pick up TM amplitude, and vice versa. Different patterns produce different mixing angles.

A general 2×2 unitary has 4 real degrees of freedom (3 if you ignore a global phase), and any one of them is reachable by a sequence "rotate–retard–rotate–...". The inverse-designed blob is doing exactly this, just continuously and all at once: every position in the blob has some local refractive-index profile that mixes and phase-shifts the modes, and the cumulative effect from input to output is the desired U.

**Big idea:** instead of building three separate components (rotator, retarder, rotator) and chaining them, we let the optimizer find one **monolithic** pattern that does everything in one shot. The result is more compact, but uninterpretable as discrete blocks — it's just a pattern that "happens to" implement the target U.

## 4. The inverse-design loop — how the pattern gets found

This is the heart of the notebook. Three pieces:

### (a) Parameters → device
The optimizer's state is a 2D array `params[i,j] ∈ [0,1]` of size 414 × 213 ≈ 88k pixels (one per 30 nm cell in the design region). To turn `params` into a physical device:

1. **Interface buffer**: force ρ=1 in a horizontal Si strip at both the left and right edges, so the input and output waveguides always butt-couple cleanly into solid Si — the optimizer can't accidentally disconnect them.
2. **Filter + project (twice)**: a 100 nm conic blur followed by a `tanh(β·(ρ−½))` projection. The blur enforces minimum feature size (no single-pixel "fuzz" that fabrication can't reproduce); the projection drives every pixel toward 0 or 1. The sharpness β is annealed: small β early (so the design is smooth and easy to optimize) and large β late (so the final design is crisply binary).
3. **Rescale**: map [0,1] → [ε_air, ε_Si] = [1.0, 12.1].

The output is a 3D permittivity map, extruded through the 220 nm Si layer thickness, plugged into Tidy3D as a `CustomMedium`.

### (b) Device → 2×2 unitary
To **measure** what U the current device implements:

- **Simulation 1**: shine pure TE into the input. Record the complex amplitudes `(a_TE_out, a_TM_out)` at the output `ModeMonitor`. This is **column 1** of U_meas — the device's response to a TE input.
- **Simulation 2**: shine pure TM into the input. Record `(a_TE_out, a_TM_out)`. This is **column 2** of U_meas.

So **two FDTD simulations per gradient evaluation** suffice to extract the entire matrix.

### (c) Figure of merit
We want U_meas to equal U_target as closely as possible, but we don't care about an overall global phase (the device might rotate the entire state by some unimportant common factor due to propagation). The gauge-invariant fidelity is

$$F = |\,{\rm Tr}(U_{\rm target}^\dagger\, U_{\rm meas})\,|^2$$

This number is in [0, 4]; it hits 4 when `U_meas = e^{iφ} · U_target` for any phase φ, and 0 when they're orthogonal. The optimization objective is

$$J = F \;-\; \lambda_{\rm pen}\cdot {\rm penalty}(\rho)$$

where the penalty discourages unfabricable features. Maximizing J makes the device match the target *and* be manufacturable.

### (d) Gradient
The miracle of the **adjoint method** is that the entire gradient ∂J/∂params (all 88,000 partial derivatives) can be computed from just **one extra FDTD simulation per source**. Tidy3D's autograd plugin handles this transparently: `value_and_grad(obj)` wraps `obj()`, automatically schedules the adjoint sims in the cloud, and returns both J and ∇J in one call. We then take an Adam step on `params` and repeat 60 times.

## 5. The training schedule

| Phase | Iterations | β | What's happening |
|---|---|---|---|
| Discovery | 1–10 | 1 (soft) | Topology emerges. Gray pixels everywhere; the optimizer is freely exploring shape space. |
| Commitment | 11–60 | linear 1→30 | Projection sharpens. Pixels are pushed toward fully Si or fully air. The pattern becomes "fabricable." |

By iteration 60, the design is essentially binary. The optimizer has navigated from "random fog" to "specific arrangement of Si features" while never losing sight of the FOM.

## 6. What does the final device look like?

You won't know exactly until you run it — that's the point of inverse design. But you can confidently expect:

- A **non-periodic, non-symmetric** pattern. Symmetric structures don't couple TE↔TM strongly; Hadamard demands strong coupling, so the optimizer breaks symmetry. Expect a "scattered confetti" look on the input side and "interfering-wavefront-shaping" on the output side.
- Feature sizes **≥ 100 nm**. The conic filter guarantees this.
- A clear **Si waveguide stub** at both the left and right edges (the interface buffer forces it).
- A pattern that looks **completely unlike a textbook waveguide circuit**. The whole point is that the optimizer finds physics-respecting solutions that no human would draw by hand.

If you've seen the user's other grating-coupler designs in `presentation_results/`, expect a similar visual style: organic-looking blob, irregular fingers and holes, working through interference and scattering rather than periodic gratings.

## 7. Verification — proving it works

Three checks at the end of the notebook:

1. **Direct matrix readout (§17).** Re-run the two-input experiment on the binarized device, build U_meas, print
   - `|U_target − U_meas|` element-wise (should be small)
   - Fidelity F (should be ≈ 1)
   - Transmission T = ‖U_meas‖²/2 (should be close to 1 — meaning little power lost to scattering or radiation)
   - Unitarity residual ‖U_meas†U_meas − I‖ (should be small — a passive lossless device is automatically unitary, so a large residual means the device leaks power)

2. **State-by-state test (§18).** Pick six canonical input states (TE, TM, the two diagonals (TE±TM)/√2, the two circulars (TE±i·TM)/√2), send each one through the FDTD simulation with both modes driven simultaneously at the right amplitude and phase, and compare the output to the analytical prediction `U_target · ψ_in`. Per-state fidelity ≈ 1 confirms the device works as a gate on *every* input, not just the two basis states the optimizer happened to train on.

3. **Cross-check against the analytical decomposition (§19).** Pure math: show that `R(−π/4) · D(π) · R(π/4)` equals the Hadamard up to a global phase. This is the "rudimentary optical circuit" from the project statement. The inverse-designed monolith and the analytical R-D-R cascade are *equivalent operations* — the notebook produces the same answer two different ways, by physics and by math.

## 8. The goal, restated

**The goal is to demonstrate that a single piece of silicon, found by topology optimization, can implement any prescribed unitary operation on the polarization-encoded state of light in a waveguide.**

Doing this once, for the Hadamard, suffices as a proof of concept. The exact same notebook with `U_target = pauli_x()` would design a TE↔TM swap gate; with `U_target = some_arbitrary_unitary()` it would design that. The framework is universal — you change one matrix at the top, you get a different device.

Why is this interesting beyond the demo?
- **Polarization is a free degree of freedom in any waveguide.** Most silicon-photonic circuits ignore TM. Treating (TE, TM) as a qubit basis doubles the information density of every waveguide.
- **No moving parts, no electrodes, no tuning.** The unitary is "frozen into" the geometry. For a fixed function this is a feature.
- **Composable.** Concatenate two of these to get a deeper circuit. Drop a phase shifter between them for reconfigurability. Build mode-multiplexed quantum circuits this way.

So the headline: this notebook is a **proof that inverse design can compile an arbitrary 2×2 unitary into ~70 µm² of patterned silicon**, with the result verified against an analytical target and against an analytical decomposition into rotators and retarders.
