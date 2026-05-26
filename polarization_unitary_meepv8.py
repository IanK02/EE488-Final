#!/usr/bin/env python
# coding: utf-8
"""
Inverse Design of a Polarization-Basis 2×2 Unitary Operator (MEEP)

Inverse-designs a patterned-Si region in a 220 nm SOI waveguide that
implements a target 2×2 unitary on the (TE₀, TM₀) polarization qubit.
Uses meep.adjoint for gradient computation and NLopt MMA for optimization.

Physics:
  - Input/output: 220 nm × 500 nm Si strip on SiO₂ BOX, air cladding
  - Design region: patterned Si slab (air/Si pixels), optimized topology
  - Two FDTD+adjoint sims per iteration (TE input, TM input) → full 2×2 U_meas
  - Objective: |Tr(U_target† U_meas)|² / 4  (gauge-invariant trace fidelity)

Setup (WSL/Ubuntu):
  conda install -c conda-forge pymeep=*=mpi_mpich_* nlopt scipy matplotlib
"""

# =============================================================================
# Section 1 — Imports
# =============================================================================
import os
import pickle
import warnings
from datetime import datetime

import autograd.numpy as anp
import matplotlib.pyplot as plt
import nlopt
import numpy as np
import scipy.ndimage as ndi
import scipy.signal as ssg

import meep as mp
import meep.adjoint as mpa

warnings.filterwarnings("ignore")
verbosity = 1
mp.verbosity(verbosity)

if mp.am_master():
    print(f"MEEP  version : {mp.__version__}")
    print(f"NumPy version : {np.__version__}")

START_TIME = datetime.now()


# =============================================================================
# Section 2 — Parameters  (all lengths in µm; 1 MEEP unit = 1 µm)
# =============================================================================

# --- Wavelength ---
wl   = 1.55           # µm
fcen = 1.0 / wl       # MEEP natural frequency

# --- Materials ---
n_Si   = 3.476
n_SiO2 = 1.444
eps_Si   = n_Si**2
eps_SiO2 = n_SiO2**2
eps_air  = 1.0

Si   = mp.Medium(epsilon=eps_Si)
SiO2 = mp.Medium(epsilon=eps_SiO2)
Air  = mp.Medium(epsilon=eps_air)

# --- Waveguide cross-section (standard 220 nm SOI) ---
t_Si  = 0.22          # Si layer thickness
w_wg  = 0.50          # waveguide width (supports both TE₀ and TM₀)
t_BOX = 1.0           # SiO₂ BOX thickness

# --- Design region ---
dr_lx = 3.0           # design region length in x
dr_ly = 2.0           # design region width in y

# --- Run config ---
RUN_NAME     = "run10_test"   # change for each run; controls output directory
RESTART_FROM = None             # path to a saved final_params.npy to warm-start

RUN_DIR          = f"misc/runs/{RUN_NAME}"
history_fname    = f"{RUN_DIR}/history.pkl"
final_params_fname = f"{RUN_DIR}/final_params.npy"
final_design_csv = f"{RUN_DIR}/final_design.csv"
os.makedirs(RUN_DIR, exist_ok=True)

# --- Optimization schedule ---
global_res  = 14      # FDTD resolution (pixels/µm); Si slab = t_Si*global_res cells thick
dr_res      = global_res
opt_steps   = 5       # total NLopt iterations

beta_min    = 1.0     # tanh projection sharpness: start soft
beta_max    = 1.0     # tanh projection sharpness: end hard (set >1 for binarization ramp)
#   beta schedule: hold beta_min for first 20% of iterations, then linearly ramp to beta_max
beta_ramp_start = int(opt_steps * 0.20)

#   fabrication penalty weight: held at 0 for first 40% of iterations, then ramped
lambda_pen_max = 0.0

# --- Source bandwidth ---
# Narrower df → more accurate single-frequency amplitudes; wider → faster sim convergence.
# Stage guide: stage1=0.05, stage2=0.035, stage3=0.025, polish=0.02
df = 0.08 * fcen

# --- Geometry ---
stub_len    = 1.0     # waveguide stub length on each side of design region
dpml        = 1.0     # PML thickness
min_feature = 0.10    # minimum feature size (µm); sets conic filter radius
filter_R    = min_feature

# --- Calibration ---
# Set True to skip the two-sim transmission calibration (saves time, disables Tphys metric)
skip_transmission_calibration = True

# =============================================================================
# Section 3 — Derived geometry (pixel-snapped)
# =============================================================================
# All physical dimensions are snapped to integer multiples of 1/global_res to
# ensure sources, monitors, and geometry boundaries fall exactly on grid edges.

def snap_len(x, res=global_res):
    """Round a length to the nearest grid pixel boundary."""
    return round(x * res) / res

def snap_pix(x, res=global_res):
    """Return the integer pixel count for a physical length."""
    return int(round(x * res))

def snap_coord(x, res=global_res):
    """Snap a coordinate to the simulation grid."""
    return round(x * res) / res

# Design region pixel counts and snapped dimensions
Nx   = snap_pix(dr_lx, global_res)
Ny   = snap_pix(dr_ly, global_res)
dr_lx = Nx / global_res
dr_ly = Ny / global_res
Npar  = Nx * Ny

# Layer thicknesses (snapped so geometry lines up with the grid)
t_Si_sim  = snap_len(t_Si,  global_res)
t_BOX_sim = snap_len(t_BOX, global_res)
z_si_bot  = -t_Si_sim / 2
z_box_bot =  z_si_bot - t_BOX_sim

# Simulation cell
stub_len = snap_len(stub_len, global_res)
dpml     = snap_len(dpml,     global_res)
sx = snap_len(dr_lx + 2 * stub_len + 2 * dpml, global_res)
sy = snap_len(dr_ly + 2 * dpml,                global_res)
sz = snap_len(t_Si_sim + t_BOX_sim + 2 * dpml, global_res)
cell = mp.Vector3(sx, sy, sz)

# Source and monitor positions (centered in the waveguide stubs)
src_x = snap_coord(-(dr_lx / 2 + stub_len / 2), global_res)
mon_x = snap_coord( (dr_lx / 2 + stub_len / 2), global_res)
mon_y = dr_ly
mon_z = snap_len(t_Si_sim + t_BOX_sim + 0.4, global_res)

# Interface-buffer dimensions: strip of forced rho=1 at left/right design edges
# ensures the optimizer never disconnects the input/output waveguide stubs.
n_border_x  = max(2, int(round(0.16 * dr_res)))
n_wg_half_y = max(1, int(round((w_wg / 2) * dr_res)))
n_cy        = Ny // 2

if mp.am_master():
    os.makedirs("misc", exist_ok=True)
    print(f"Design region : {dr_lx} µm × {dr_ly} µm  ({Nx}×{Ny} = {Npar:,} params)")
    print(f"Resolution    : {global_res} px/µm  (Si slab = {t_Si_sim*global_res:.1f} cells thick)")
    print(f"Cell size     : {sx:.3f} × {sy:.3f} × {sz:.3f} µm")
    print(f"src_x={src_x:.3f} µm,  mon_x={mon_x:.3f} µm")


# =============================================================================
# Section 4 — Static geometry (BOX + waveguide stubs)
# =============================================================================
# The design-region block is built separately in each OptimizationProblem so
# the MaterialGrid weights can be updated without touching these structures.

def static_geometry():
    """SiO₂ BOX slab + input and output Si waveguide stubs."""
    INF = 1e6
    stub_size_x = sx / 2 - dr_lx / 2 + 0.02   # slight overlap into design boundary

    box_layer = mp.Block(
        size     = mp.Vector3(INF, INF, t_BOX_sim),
        center   = mp.Vector3(0, 0, z_si_bot - t_BOX_sim / 2),
        material = SiO2,
    )
    wg_in = mp.Block(
        size     = mp.Vector3(stub_size_x, w_wg, t_Si_sim),
        center   = mp.Vector3(-(dr_lx / 2 + stub_size_x / 2 - 0.01), 0, 0),
        material = Si,
    )
    wg_out = mp.Block(
        size     = mp.Vector3(stub_size_x, w_wg, t_Si_sim),
        center   = mp.Vector3( (dr_lx / 2 + stub_size_x / 2 - 0.01), 0, 0),
        material = Si,
    )
    return [box_layer, wg_in, wg_out]


# =============================================================================
# Section 5 — Preprocessing pipeline (filter → project, applied twice)
# =============================================================================
# Converts raw optimizer parameters p ∈ [0,1]^Npar into a physical density ρ:
#   1. interface_buffer  : force ρ=1 in a w_wg-wide strip at left/right edges
#   2. conic_filter      : spatial blur enforcing ≥ min_feature minimum feature size
#   3. tanh_proj(β)      : drive pixels toward 0 or 1; sharpness increases with β
# Steps 2–3 are applied twice for cleaner binarization.
#
# pre_process_grad() analytically back-propagates dJ/dρ → dJ/dp through this
# same pipeline (chain rule), which the MEEP adjoint gradient requires.

def _make_conic_kernel(radius_um, res_pxum):
    """Conic (linearly decaying) blur kernel for minimum-feature-size enforcement."""
    r_px = int(np.ceil(radius_um * res_pxum))
    sz_k = 2 * r_px + 1
    k    = np.zeros((sz_k, sz_k))
    for i in range(sz_k):
        for j in range(sz_k):
            d = np.sqrt((i - r_px)**2 + (j - r_px)**2) / (r_px + 1e-9)
            if d <= 1.0:
                k[i, j] = 1.0 - d
    k /= k.sum()
    return k

_conic_k = _make_conic_kernel(filter_R, dr_res)

def conic_filter(rho2d):
    return ssg.convolve2d(rho2d, _conic_k, mode="same", boundary="fill", fillvalue=0.0)

def tanh_proj(rho2d, beta):
    """Smooth threshold: maps [0,1] → [0,1] with sharpness β. β→∞ gives a step."""
    t = np.tanh(beta * 0.5)
    return 0.5 + 0.5 * np.tanh(beta * (rho2d - 0.5)) / t

def dtanh_proj(rho2d, beta):
    """Elementwise derivative of tanh_proj w.r.t. rho2d (for backprop)."""
    t = np.tanh(beta * 0.5)
    return 0.5 * beta * (1.0 - np.tanh(beta * (rho2d - 0.5))**2) / t

def interface_buffer(p2d):
    """Force ρ=1 in the waveguide-width strip at the left and right design edges."""
    p  = p2d.copy()
    y0 = n_cy - n_wg_half_y
    y1 = n_cy + n_wg_half_y + 1
    p[0:n_border_x,          y0:y1] = 1.0
    p[Nx - n_border_x:Nx,    y0:y1] = 1.0
    return p

def pre_process(params_flat, beta):
    """params_flat (Npar,) → physical density ρ (Npar,), values in [0,1]."""
    p = params_flat.reshape(Nx, Ny)
    p = interface_buffer(p)
    p = tanh_proj(conic_filter(p), beta)
    p = tanh_proj(conic_filter(p), beta)
    return p.ravel()

def pre_process_grad(params_flat, grad_rho_flat, beta):
    """Back-propagate dJ/dρ → dJ/dp through the preprocessing pipeline."""
    p0 = interface_buffer(params_flat.reshape(Nx, Ny))
    f1 = conic_filter(p0)
    p1 = tanh_proj(f1, beta)
    f2 = conic_filter(p1)

    g = grad_rho_flat.reshape(Nx, Ny)
    g = g * dtanh_proj(f2, beta)                                          # ← 2nd tanh
    g = ssg.convolve2d(g, _conic_k, mode="same", boundary="fill", fillvalue=0.0)  # ← 2nd filter
    g = g * dtanh_proj(f1, beta)                                          # ← 1st tanh
    g = ssg.convolve2d(g, _conic_k, mode="same", boundary="fill", fillvalue=0.0)  # ← 1st filter

    # Interface-buffer pixels are not free variables; zero their gradient
    y0, y1 = n_cy - n_wg_half_y, n_cy + n_wg_half_y + 1
    g[0:n_border_x,       y0:y1] = 0.0
    g[Nx-n_border_x:Nx,   y0:y1] = 0.0
    return g.ravel()

if mp.am_master():
    print(f"Conic kernel : {_conic_k.shape[0]}×{_conic_k.shape[1]} px  "
          f"(radius={filter_R} µm @ {dr_res} px/µm)")


# =============================================================================
# Section 6 — Fabrication penalty (erosion-dilation, Lazarov et al. 2016)
# =============================================================================
# penalty = mean( ρ_eroded · (1 − ρ_dilated) )
#   = 0   when every pixel is 0 or 1 (fully binary)
#   > 0   when gray pixels exist
# Smooth erosion/dilation approximations allow gradient computation.

def conic_filter_T(g):
    """Adjoint of conic_filter. Symmetric kernel → same convolution."""
    return ssg.convolve2d(g, _conic_k, mode="same", boundary="fill", fillvalue=0.0)

def fab_penalty_and_grad(params_flat, beta, beta_ed=8.0):
    """Fabrication penalty and its gradient w.r.t. raw params_flat."""
    rho = pre_process(params_flat, beta).reshape(Nx, Ny)

    # Erosion branch: ρ_ero = 1 − tanh_proj(conic_filter(1 − ρ))
    q        = 1.0 - rho
    fq       = conic_filter(q)
    rho_ero  = 1.0 - tanh_proj(fq, beta_ed)

    # Dilation branch: ρ_dil = tanh_proj(conic_filter(ρ))
    fr       = conic_filter(rho)
    rho_dil  = tanh_proj(fr, beta_ed)

    penalty  = float(np.mean(rho_ero * (1.0 - rho_dil)))
    scale    = 1.0 / (Nx * Ny)

    # Backprop erosion
    g_fq           = (-scale * (1.0 - rho_dil)) * dtanh_proj(fq, beta_ed)
    g_rho_from_ero = -conic_filter_T(g_fq)          # minus sign from (1 − ρ) input

    # Backprop dilation
    g_fr           = (-scale * rho_ero) * dtanh_proj(fr, beta_ed)
    g_rho_from_dil =  conic_filter_T(g_fr)

    g_rho    = g_rho_from_ero + g_rho_from_dil
    g_params = pre_process_grad(params_flat, g_rho.ravel(), beta)
    return penalty, g_params


# =============================================================================
# Section 7 — Mode calibration
# =============================================================================
# MEEP sorts eigenmodes by descending Re(neff).
# For 220 nm × 500 nm Si-on-SiO₂ at 1550 nm:
#   band 1 → TE₀  (Ey-dominant, neff ≈ 2.4)
#   band 2 → TM₀  (Ez-dominant, neff ≈ 1.7)
# We verify empirically by sampling |Ey|² vs |Ez|² inside the waveguide core.

eig_band_TE = 1
eig_band_TM = 2

def calibrate_modes():
    """Assign eig_band_TE / eig_band_TM by polarization fraction."""
    global eig_band_TE, eig_band_TM

    wg_geom = [
        mp.Block(size=mp.Vector3(1e6, 1e6, t_BOX_sim),
                 center=mp.Vector3(0, 0, z_si_bot - t_BOX_sim / 2), material=SiO2),
        mp.Block(size=mp.Vector3(1e6, w_wg, t_Si_sim),
                 center=mp.Vector3(0, 0, 0), material=Si),
    ]
    sim_cal = mp.Simulation(
        cell_size       = mp.Vector3(0.2, sy, sz),
        boundary_layers = [mp.PML(dpml, direction=mp.Y), mp.PML(dpml, direction=mp.Z)],
        geometry        = wg_geom,
        sources         = [mp.EigenModeSource(
            src=mp.GaussianSource(fcen, fwidth=df * 10),
            center=mp.Vector3(0, 0, 0), size=mp.Vector3(0, mon_y, mon_z), eig_band=1,
        )],
        resolution=global_res,
    )
    sim_cal.init_sim()

    mon_vol = mp.Volume(center=mp.Vector3(0, 0, 0), size=mp.Vector3(0, mon_y, mon_z))
    results = {}
    for band in [1, 2]:
        em  = sim_cal.get_eigenmode(fcen, mp.X, mon_vol, band, mp.Vector3(1, 0, 0))
        ys  = np.linspace(-w_wg / 2 * 0.9,  w_wg / 2 * 0.9,  8)
        zs  = np.linspace(-t_Si  / 2 * 0.9, t_Si  / 2 * 0.9, 4)
        Ey2, Ez2 = 0.0, 0.0
        for y in ys:
            for z in zs:
                pt   = mp.Vector3(0, y, z)
                Ey2 += abs(em.amplitude(pt, mp.Ey))**2
                Ez2 += abs(em.amplitude(pt, mp.Ez))**2
        te_frac       = Ey2 / (Ey2 + Ez2 + 1e-30)
        neff          = em.kdom.x / fcen
        results[band] = (te_frac, neff)
        if mp.am_master():
            print(f"  band {band}: neff={neff:.4f}  |Ey|² frac={te_frac:.3f}")

    eig_band_TE = 1 if results[1][0] >= results[2][0] else 2
    eig_band_TM = 3 - eig_band_TE
    if mp.am_master():
        print(f"  → eig_band_TE={eig_band_TE},  eig_band_TM={eig_band_TM}")
    return eig_band_TE, eig_band_TM

if mp.am_master():
    print("Running mode calibration…")
eig_band_TE, eig_band_TM = calibrate_modes()


# =============================================================================
# Section 8 — Target unitary
# =============================================================================
# U_target is the 2×2 unitary we want the device to implement on (TE₀, TM₀).
# Change this one line to design for a different gate; delete history.pkl first.

def hadamard():
    return (1.0 / np.sqrt(2)) * np.array([[1,  1], [1, -1]], dtype=complex)

def pauli_x():
    return np.array([[0, 1], [1, 0]], dtype=complex)

def pauli_z():
    return np.array([[1, 0], [0, -1]], dtype=complex)

def rotator(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[c, -s], [s, c]], dtype=complex)

def retarder(phi):
    return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)

def is_unitary(U, tol=1e-9):
    return np.allclose(U.conj().T @ U, np.eye(2), atol=tol)

U_target = hadamard()
assert is_unitary(U_target), "U_target is not unitary!"

# Reference normalization: populated by calibrate_transmission_reference()
# to convert raw MEEP amplitudes to physical transmission fractions.
REF_NORM = {"TE": None, "TM": None}

if mp.am_master():
    print("U_target =")
    print(np.round(U_target, 4))


# =============================================================================
# Section 9 — OptimizationProblem objects (one per input polarization)
# =============================================================================
# Pattern:
#   design_vars = mp.MaterialGrid(...)       weights ∈ [0,1]: 0=Air, 1=Si
#   dr          = mpa.DesignRegion(...)
#   te_mon      = mpa.EigenmodeCoefficient(...)  complex amplitude of TE₀ at output
#   tm_mon      = mpa.EigenmodeCoefficient(...)  complex amplitude of TM₀ at output
#   opt([rho])  → (f_val, (dJ_dweights,))
#
# Objective per column:
#   J_col = |conj(U_target[:, col]) · [a_te, a_tm]|²
#
# Summing both columns: J_TE + J_TM ≈ |Tr(U_target† U_meas)|²  (up to cross terms).
# Gradients from the two adjoint solves are combined in evaluate_and_grad().

# ------------------------------------------------------------
# Dynamic objective controls for exact F_trace gradient
# ------------------------------------------------------------
# Objective modes:
#   "probe"      : returns a harmless scalar; used to measure amplitudes.
#                  Gradients from this pass are ignored.
#
#   "trace_grad" : returns 0.5 * Re(conj(S) * s_col),
#                  where S = s_TE + s_TM is the current full trace overlap.
#                  Summing TE + TM gradients gives the exact current
#                  F_trace gradient:
#
#                    F_trace = |S|^2 / 4
#                    dF = 0.5 * Re(conj(S) dS)

DYN_OBJ = {
    "mode": "probe",
    "trace_weight": 1.0 + 0.0j,
}


def _make_opt(source_band, col_idx):
    """
    Build a persistent OptimizationProblem for one input polarization column.

    Args:
        source_band : MEEP eig_band for the input mode (eig_band_TE or eig_band_TM)
        col_idx     : column index in U_target (0 for TE input, 1 for TM input)
    """
    design_vars = mp.MaterialGrid(
        mp.Vector3(Nx, Ny, 1),
        Air, Si,
        weights   = np.full(Npar, 0.5),
        grid_type = "U_DEFAULT",
    )
    design_block = mp.Block(
        size     = mp.Vector3(dr_lx, dr_ly, t_Si_sim),
        center   = mp.Vector3(0, 0, 0),
        material = design_vars,
    )
    dr = mpa.DesignRegion(
        design_parameters = design_vars,
        volume = mp.Volume(center=mp.Vector3(0, 0, 0),
                           size=mp.Vector3(dr_lx, dr_ly, t_Si_sim)),
    )
    src = mp.EigenModeSource(
        src            = mp.GaussianSource(fcen, fwidth=df),
        center         = mp.Vector3(src_x, 0, 0),
        size           = mp.Vector3(0, mon_y, mon_z),
        eig_band       = source_band,
        eig_match_freq = True,
        eig_parity     = mp.NO_PARITY,
        direction      = mp.X,
    )
    sim = mp.Simulation(
        cell_size       = cell,
        boundary_layers = [mp.PML(dpml)],
        geometry        = static_geometry() + [design_block],
        sources         = [src],
        resolution      = global_res,
        eps_averaging   = False,
    )
    mon_vol = mp.Volume(center=mp.Vector3(mon_x, 0, 0),
                        size=mp.Vector3(0, mon_y, mon_z))
    te_mon = mpa.EigenmodeCoefficient(sim, mon_vol, mode=eig_band_TE,
                                      eig_parity=mp.NO_PARITY, forward=True)
    tm_mon = mpa.EigenmodeCoefficient(sim, mon_vol, mode=eig_band_TM,
                                      eig_parity=mp.NO_PARITY, forward=True)

    # Capture col_idx in closure so each opt object has its own fixed column target.
    _col = col_idx
    def objective(a_te, a_tm):
        """
        Dynamic objective.

        In probe mode:
            Used only to get amplitudes. Gradient is ignored.

        In trace_grad mode:
            Returns the exact local objective whose gradient equals the
            current-step contribution to:

                F_trace = |Tr(U_target† U_meas)|² / 4

            using the trace weight S computed from the current measured matrix.
        """

        t0 = anp.conj(U_target[0, _col])
        t1 = anp.conj(U_target[1, _col])

        # Raw column overlap:
        #   s_col = <target_col | measured_col>
        raw_overlap = t0 * a_te[0] + t1 * a_tm[0]

        # Column-normalized overlap.
        # This matches your current gate metric, where each measured output column
        # is normalized before computing F_trace.
        col_norm = anp.sqrt(anp.abs(a_te[0])**2 + anp.abs(a_tm[0])**2 + 1e-30)
        overlap = raw_overlap / col_norm

        if DYN_OBJ["mode"] == "probe":
            # Harmless scalar. We ignore this gradient.
            return anp.abs(overlap) ** 2

        elif DYN_OBJ["mode"] == "trace_grad":
            S = complex(DYN_OBJ["trace_weight"])

            # Exact differential:
            #   F = |S|² / 4
            #   dF = 0.5 * Re(conj(S) dS)
            #
            # Each column contributes dS_col, so each OptimizationProblem returns:
            #   0.5 * Re(conj(S) * s_col)
            return 0.5 * anp.real(anp.conj(S) * overlap)

        else:
            raise ValueError(f"Unknown DYN_OBJ mode: {DYN_OBJ['mode']}")

    return mpa.OptimizationProblem(
        simulation          = sim,
        objective_functions = [objective],
        objective_arguments = [te_mon, tm_mon],
        design_regions      = [dr],
        frequencies         = [fcen],
    )

if mp.am_master():
    print("Building TE optimization problem…")
opt_TE = _make_opt(eig_band_TE, col_idx=0)

if mp.am_master():
    print("Building TM optimization problem…")
opt_TM = _make_opt(eig_band_TM, col_idx=1)

if mp.am_master():
    print("OptimizationProblem objects ready.")


# =============================================================================
# Section 10 — Forward/adjoint helper and optional transmission calibration
# =============================================================================

def _run_forward_only(opt_obj, rho):
    """
    Run forward-only pass for amplitude probing, avoiding adjoint gradient work.
    """
    try:
        result = opt_obj([rho], need_gradient=False)
    except TypeError:
        result = opt_obj([rho], need_value=True, need_gradient=False)

    args = opt_obj.get_objective_arguments()
    a_te = complex(args[0][0])
    a_tm = complex(args[1][0])

    if isinstance(result, (tuple, list)):
        f_val = result[0]
    else:
        f_val = result

    return float(np.real(f_val)), a_te, a_tm

def _run_adjoint(opt_obj, rho):
    """
    Run one MEEP forward+adjoint pass.

    Returns:
        f_val    : scalar objective value
        grad_arr : gradient w.r.t. MaterialGrid weights, shape (Npar,)
        a_te     : complex TE₀ output amplitude at the monitor
        a_tm     : complex TM₀ output amplitude at the monitor
    """
    result   = opt_obj([rho])
    f_val    = result[0]
    grad_arr = result[1]
    args     = opt_obj.get_objective_arguments()
    a_te     = complex(args[0][0])   # single-frequency result
    a_tm     = complex(args[1][0])
    return float(np.real(f_val)), grad_arr, a_te, a_tm


def calibrate_transmission_reference(beta=1.0):
    """
    Measure output amplitudes for a straight waveguide in a properly bounded cell.
    """
    global REF_NORM
    if mp.am_master():
        print("Calibrating transmission reference (Using bounded fast calibration cell)...")

    # Set verbosity to 0 (Complete Silence)
    # mp.verbosity(0)

    # (Your existing geometry configuration setup...)
    wg_geom = [
        mp.Block(size=mp.Vector3(1e6, 1e6, t_BOX_sim),
                 center=mp.Vector3(0, 0, z_si_bot - t_BOX_sim / 2), material=SiO2),
        mp.Block(size=mp.Vector3(1e6, w_wg, t_Si_sim),
                 center=mp.Vector3(0, 0, 0), material=Si),
    ]

    sim_cal = mp.Simulation(
        cell_size       = cell,
        boundary_layers = [mp.PML(dpml)], 
        geometry        = wg_geom,
        sources         = [mp.EigenModeSource(
            src            = mp.GaussianSource(fcen, fwidth=df),
            center         = mp.Vector3(src_x, 0, 0),
            size           = mp.Vector3(0, mon_y, mon_z),
            eig_band       = eig_band_TE,
            eig_match_freq = True,
            direction      = mp.X,
        )],
        resolution=global_res,
        eps_averaging=False,
    )

    mon_vol = mp.Volume(center=mp.Vector3(mon_x, 0, 0), size=mp.Vector3(0, mon_y, mon_z))
    
    # Run TE calibration in complete silence
    sim_cal.init_sim()
    sim_cal.run(
        mp.at_every(20, mp.stop_when_fields_decayed(1e-5, mp.Ey, mp.Vector3(mon_x, 0, 0), 1e-3)), 
        until_after_sources=300
    )
    em_te = sim_cal.get_eigenmode(fcen, mp.X, mon_vol, eig_band_TE, mp.Vector3(1, 0, 0))
    REF_NORM["TE"] = complex(em_te.amplitude(mp.Vector3(mon_x, 0, 0), mp.Ey))

    # Reset and run TM calibration in complete silence
    sim_cal.reset_meep()
    sim_cal.sources = [mp.EigenModeSource(
        src            = mp.GaussianSource(fcen, fwidth=df),
        center         = mp.Vector3(src_x, 0, 0),
        size           = mp.Vector3(0, mon_y, mon_z),
        eig_band       = eig_band_TM,
        eig_match_freq = True,
        direction      = mp.X,
    )]
    sim_cal.init_sim()
    sim_cal.run(
        mp.at_every(20, mp.stop_when_fields_decayed(1e-5, mp.Ez, mp.Vector3(mon_x, 0, 0), 1e-3)), 
        until_after_sources=300
    )
    em_tm = sim_cal.get_eigenmode(fcen, mp.X, mon_vol, eig_band_TM, mp.Vector3(1, 0, 0))
    REF_NORM["TM"] = complex(em_tm.amplitude(mp.Vector3(mon_x, 0, 0), mp.Ez))

    # Restore original verbosity so you still see logs during optimization iterations
    mp.verbosity(verbosity)

    if mp.am_master():
        print(f"  → CALIBRATION DONE: TE_Ref={REF_NORM['TE']:.4f}, TM_Ref={REF_NORM['TM']:.4f}")


# =============================================================================
# Section 11 — Initial design
# =============================================================================
# Strategy:
# Build a three-zone warm start: rotator | retarder | rotator
# Each zone is a rough approximation — the optimizer refines from here.

np.random.seed(42)
init_2d = np.zeros((Nx, Ny))

x_coords = np.linspace(-dr_lx/2, dr_lx/2, Nx)
y_coords = np.linspace(-dr_ly/2, dr_ly/2, Ny)
XX, YY = np.meshgrid(x_coords, y_coords, indexing='ij')

# Zone boundaries (in x)
z1 = dr_lx * 0.25   # first rotator ends here
z2 = dr_lx * 0.75   # retarder ends here

# Zone 1: rotator — diagonal Si stripes at 45 degrees
# Angled boundaries scatter TE<->TM
stripe_pitch = w_wg * 1.2
in_zone1 = XX < (-dr_lx/2 + z1)
stripe1 = 0.50 + 0.25 * np.sin(2 * np.pi * (XX + YY) / stripe_pitch)
init_2d[in_zone1] = stripe1[in_zone1]

# Zone 2: retarder — widened waveguide to accumulate differential phase
# Fill with Si in a wider-than-nominal strip to shift neff(TE) - neff(TM)
in_zone2 = (XX >= (-dr_lx/2 + z1)) & (XX < (-dr_lx/2 + z2))
wider_wg = (np.abs(YY) < w_wg * 1.5).astype(float)
init_2d[in_zone2] = wider_wg[in_zone2]

# Zone 3: second rotator — diagonal stripes, opposite orientation
in_zone3 = XX >= (-dr_lx/2 + z2)
stripe3 = 0.5 + 0.25 * np.sin(2 * np.pi * (XX - YY) / stripe_pitch)
init_2d[in_zone3] = stripe3[in_zone3]

# Add small noise to break degeneracy within each zone
noise = np.random.uniform(-0.15, 0.15, (Nx, Ny))
init_2d = np.clip(init_2d + noise, 0, 1).ravel()

init_par = init_2d

# Visualize initial density
rho_init = pre_process(init_par, beta=beta_min).reshape(Nx, Ny)
eps_init = eps_air + (eps_Si - eps_air) * rho_init

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)
im = ax0.imshow(eps_init.T, extent=[-dr_lx/2, dr_lx/2, -dr_ly/2, dr_ly/2],
                origin="lower", cmap="RdGy_r", vmin=eps_air, vmax=eps_Si)
plt.colorbar(im, ax=ax0, label="ε")
ax0.set_title("Initial design (xy, z=0)")
ax0.set_xlabel("x (µm)"); ax0.set_ylabel("y (µm)")
ax1.hist(rho_init.ravel(), bins=50, color="steelblue", edgecolor="none")
ax1.set_xlabel("Density ρ"); ax1.set_ylabel("Count")
ax1.set_title("Initial density histogram")
if mp.am_master():
    plt.savefig(f"{RUN_DIR}/initial_design.png", dpi=200)
    plt.close()
    print(f"init_par ∈ [{init_par.min():.3f}, {init_par.max():.3f}]")


# =============================================================================
# Section 12 — Checkpoint management
# =============================================================================

def save_history(h):
    if mp.am_master():
        with open(history_fname, "wb") as fh:
            pickle.dump(h, fh)
    mp.comm.barrier()

def load_history():
    with open(history_fname, "rb") as fh:
        return pickle.load(fh)

def resize_params_if_needed(p_old, old_shape, new_shape):
    """Bilinearly interpolate params from one grid resolution to another."""
    if old_shape == new_shape:
        return p_old.copy()
    p2d   = p_old.reshape(old_shape)
    p_new = ndi.zoom(p2d, (new_shape[0] / old_shape[0], new_shape[1] / old_shape[1]), order=1)
    return np.clip(p_new, 0.0, 1.0).ravel()
    
_history_keys = [
    "J", "params", "grad", "beta", "U_meas",
    "Fh", "F_trace", "F_col", "F_TE", "F_TM",
    "Tavg", "Uerr", "Penalty",
    "balance_pen", "mix_ratio",
]

try:
    history  = load_history()
    params   = history["params"][-1].copy()
    num_done = len(history["J"])
    if mp.am_master():
        print(f"Loaded checkpoint: {num_done}/{opt_steps} iterations done.")
except FileNotFoundError:
    if RESTART_FROM is not None:
        ckpt   = np.load(RESTART_FROM, allow_pickle=True).item()
        params = resize_params_if_needed(ckpt["params"], tuple(ckpt["shape"]), (Nx, Ny))
        if mp.am_master():
            print(f"Restarting from {RESTART_FROM}  (shape {ckpt['shape']} → {(Nx,Ny)})")
    else:
        params = init_par.copy()
        if mp.am_master():
            print("No checkpoint found — starting fresh.")
    history = {k: [] for k in _history_keys}


# =============================================================================
# Section 13 — Main evaluation: fidelity, gradient, diagnostics
# =============================================================================
# Two FDTD+adjoint simulations per call (TE input, TM input).
#
# Objective (trace fidelity):
#   J = |Tr(U_target† U_meas)|² / 4  −  λ_pen · fabrication_penalty
#   range [0, 1]; J=1 when U_meas = e^{iφ} U_target (gauge-invariant)
#
# Gradient assembly:
#   Each adjoint solve returns dJ_col/d(MaterialGrid weights).
#   The 0.25 factor matches the 1/4 normalization in the trace fidelity.
#   pre_process_grad() back-propagates through filter+project to raw params.

def evaluate_and_grad(params_flat, beta, lambda_pen_val, step_num=None):
    """
    Exact current-step F_trace gradient version.

    Cost:
        4 MEEP adjoint calls per optimization iteration:
          1. TE probe pass
          2. TM probe pass
          3. TE trace-gradient pass
          4. TM trace-gradient pass

    Why 4?
        The true trace fidelity is:

            F_trace = |s_TE + s_TM|² / 4

        The gradient needs the current S = s_TE + s_TM.
        Since S is only known after both columns are measured, we first probe
        the current columns, compute S, then rerun adjoint objectives weighted
        by that current S.
    """

    rho = pre_process(params_flat, beta)

    # ------------------------------------------------------------
    # 1. Probe pass: get current amplitudes
    # ------------------------------------------------------------
    DYN_OBJ["mode"] = "probe"

    _, a_te_TE, a_tm_TE = _run_forward_only(opt_TE, rho)
    _, a_te_TM, a_tm_TM = _run_forward_only(opt_TM, rho)

    U_raw = np.array([
        [a_te_TE, a_te_TM],
        [a_tm_TE, a_tm_TM],
    ], dtype=complex)

    # ------------------------------------------------------------
    # 2. Build measured normalized matrix
    # ------------------------------------------------------------
    norm_TE = np.sqrt(abs(a_te_TE)**2 + abs(a_tm_TE)**2 + 1e-30)
    norm_TM = np.sqrt(abs(a_te_TM)**2 + abs(a_tm_TM)**2 + 1e-30)

    u_TE = np.array([a_te_TE, a_tm_TE], dtype=complex) / norm_TE
    u_TM = np.array([a_te_TM, a_tm_TM], dtype=complex) / norm_TM

    U_meas = np.array([
        [u_TE[0], u_TM[0]],
        [u_TE[1], u_TM[1]],
    ], dtype=complex)

    # ------------------------------------------------------------
    # 3. True gate fidelity metrics
    # ------------------------------------------------------------
    s_TE = np.vdot(U_target[:, 0], u_TE)
    s_TM = np.vdot(U_target[:, 1], u_TM)

    trace_overlap = s_TE + s_TM
    F_trace = float(abs(trace_overlap)**2 / 4.0)

    F_TE = float(abs(s_TE)**2)
    F_TM = float(abs(s_TM)**2)
    F_col = float(0.5 * (F_TE + F_TM))

    Uerr = float(np.linalg.norm(U_meas.conj().T @ U_meas - np.eye(2), "fro"))
    col_overlap = np.vdot(U_meas[:, 0], U_meas[:, 1])

    # Mixing diagnostics
    Uabs = np.abs(U_meas)
    balance_pen = float(np.sum((Uabs**2 - 0.5)**2))
    offdiag_power = float(abs(U_meas[0, 1])**2 + abs(U_meas[1, 0])**2)
    diag_power = float(abs(U_meas[0, 0])**2 + abs(U_meas[1, 1])**2)
    mix_ratio = offdiag_power / (diag_power + offdiag_power + 1e-30)

    # Transmission proxy.
    # This is still uncalibrated unless you later add a reliable reference.
    Traw = float(np.linalg.norm(U_raw, "fro")**2 / 2.0)

    # ------------------------------------------------------------
    # 4. Exact current F_trace gradient pass
    # ------------------------------------------------------------
    DYN_OBJ["mode"] = "trace_grad"
    DYN_OBJ["trace_weight"] = trace_overlap

    _, grad_te, _, _ = _run_adjoint(opt_TE, rho)
    _, grad_tm, _, _ = _run_adjoint(opt_TM, rho)

    dFtrace_drho = grad_te.ravel() + grad_tm.ravel()
    grad_optical = pre_process_grad(params_flat, dFtrace_drho, beta)

    # ------------------------------------------------------------
    # 5. Fabrication penalty and final objective
    # ------------------------------------------------------------
    pen, grad_fab = fab_penalty_and_grad(params_flat, beta)

    J = F_trace - lambda_pen_val * pen
    grad = grad_optical - lambda_pen_val * grad_fab

    # ------------------------------------------------------------
    # 6. Logging
    # ------------------------------------------------------------
    tag = f"step {step_num}" if step_num is not None else "eval"

    if mp.am_master():
        print("\n" + "-" * 78, flush=True)
        print(f"[{tag}] Exact F_trace gradient | beta={beta:.3f}", flush=True)
        print("-" * 78, flush=True)

        print("OBJECTIVE", flush=True)
        print(f"  J          = {J:.6f}", flush=True)
        print(f"  F_trace    = {F_trace:.6f}   TRUE coherent gate fidelity", flush=True)
        print(f"  F_col      = {F_col:.6f}   independent-column diagnostic", flush=True)
        print(f"  F_TE       = {F_TE:.6f}", flush=True)
        print(f"  F_TM       = {F_TM:.6f}", flush=True)

        print("\nPHASE / TRACE", flush=True)
        print(f"  s_TE       = {s_TE:.6f}", flush=True)
        print(f"  s_TM       = {s_TM:.6f}", flush=True)
        print(f"  S_trace    = {trace_overlap:.6f}", flush=True)

        print("\nMIXING", flush=True)
        print("  |U_meas| =", flush=True)
        print(Uabs, flush=True)
        print(f"  balance_pen = {balance_pen:.6f}   ideal=0 for Hadamard 50/50 columns", flush=True)
        print(f"  mix_ratio   = {mix_ratio:.6f}   target≈0.5 for balanced Hadamard", flush=True)

        print("\nCONSTRAINTS", flush=True)
        print(f"  Uerr       = {Uerr:.6f}", flush=True)
        print(f"  |u0†u1|    = {abs(col_overlap):.6f}", flush=True)
        print(f"  Penalty    = {pen:.6f}", flush=True)
        print(f"  penalty_w  = {(lambda_pen_val * pen):.6f}", flush=True)
        print(f"  Traw       = {Traw:.6f}   uncalibrated", flush=True)

        print("\nMATRIX", flush=True)
        print("  U_meas =", flush=True)
        print(U_meas, flush=True)
        print("-" * 78 + "\n", flush=True)

    info = {
        "J": float(np.real(J)),

        # True primary metric
        "Fh": float(F_trace),
        "F_trace": float(F_trace),

        # Diagnostics
        "F_col": float(F_col),
        "F_TE": float(F_TE),
        "F_TM": float(F_TM),

        "Tavg": float(Traw),
        "Uerr": float(Uerr),
        "Penalty": float(pen),

        "U": U_meas.copy(),
        "U_raw": U_raw.copy(),

        "s_TE": s_TE,
        "s_TM": s_TM,
        "trace_overlap": trace_overlap,
        "col_overlap": col_overlap,

        "balance_pen": float(balance_pen),
        "mix_ratio": float(mix_ratio),
    }

    return float(np.real(J)), np.real(grad), info

# =============================================================================
# Section 14 — Optimization loop (NLopt LD_MMA)
# =============================================================================
# MMA (Method of Moving Asymptotes) is the standard algorithm for large-scale
# topology optimization with box constraints.
# NLopt convention: set_max_objective → grad[:] = dJ/dx (ascent direction).
#
# Schedules:
#   beta     : flat at beta_min for first 20% of iterations, then linear ramp to beta_max
#   λ_pen    : 0 for first 40% of iterations, then linear ramp to lambda_pen_max

iteration_counter = [len(history["J"])]

def nlopt_callback(x, grad):
    i = iteration_counter[0]
    if mp.am_master():
        print(f"\n{'='*55}\nIteration {i+1}/{opt_steps} - {datetime.now()}")

    # Beta schedule
    if i < beta_ramp_start:
        current_beta = beta_min
    else:
        frac         = (i - beta_ramp_start) / max(1, opt_steps - 1 - beta_ramp_start)
        current_beta = beta_min + (beta_max - beta_min) * min(frac, 1.0)

    # Fabrication penalty schedule
    pen_start          = int(opt_steps * 0.40)
    current_lambda_pen = 0.0 if i < pen_start else (
        lambda_pen_max * min((i - pen_start) / max(1, opt_steps - 1 - pen_start), 1.0)
    )

    J_val, grad_val, info = evaluate_and_grad(
        params_flat    = x,
        beta           = current_beta,
        lambda_pen_val = current_lambda_pen,
        step_num       = i + 1,
    )

    if grad.size > 0:
        grad[:] = np.real(grad_val)

    # Update history
    U_meas   = info["U"]
    Fh_val   = float(abs(np.trace(U_target.conj().T @ U_meas))**2 / 4.0)
    history["J"].append(float(J_val))
    history["params"].append(x.copy())
    history["grad"].append(np.real(grad_val).copy())
    history["beta"].append(float(current_beta))
    history["U_meas"].append(U_meas.copy())
    history["Fh"].append(Fh_val)
    history["Tavg"].append(info["Tavg"])
    history["Uerr"].append(info["Uerr"])
    history["Penalty"].append(info["Penalty"])

    if "F_trace" in history:
        history["F_trace"].append(float(info["F_trace"]))
    if "F_col" in history:
        history["F_col"].append(float(info["F_col"]))
    if "F_TE" in history:
        history["F_TE"].append(float(info["F_TE"]))
    if "F_TM" in history:
        history["F_TM"].append(float(info["F_TM"]))
    if "balance_pen" in history:
        history["balance_pen"].append(float(info["balance_pen"]))
    if "mix_ratio" in history:
        history["mix_ratio"].append(float(info["mix_ratio"]))

    save_history(history)
    if mp.am_master():
        np.save(final_params_fname, {
            "params":     x.copy(),
            "shape":      (Nx, Ny),
            "global_res": global_res,
            "beta":       current_beta,
            "run_name":   RUN_NAME,
        })
    mp.comm.barrier()
    iteration_counter[0] += 1
    return float(J_val)


# Optional transmission calibration (costs 2 extra FDTD sims)
if not skip_transmission_calibration:
    calibrate_transmission_reference(beta=beta_min)

# Launch NLopt
remaining = opt_steps - len(history["J"])
if remaining > 0:
    opt_nlopt = nlopt.opt(nlopt.LD_MMA, Npar)
    opt_nlopt.set_lower_bounds(0.0)
    opt_nlopt.set_upper_bounds(1.0)
    opt_nlopt.set_max_objective(nlopt_callback)
    opt_nlopt.set_maxeval(remaining)
    opt_nlopt.set_ftol_rel(1e-6)
    if mp.am_master():
        print(f"Starting NLopt MMA ({remaining} iterations remaining)…")
    params = opt_nlopt.optimize(params.copy())
    if mp.am_master():
        print(f"Optimization finished.  Final J = {opt_nlopt.last_optimum_value():.4f}")
else:
    if mp.am_master():
        print("Optimization already complete (loaded from checkpoint).")


# =============================================================================
# Section 15 — Convergence plots + final design map
# =============================================================================
J_vals      = np.array(history["J"])
Jrep_vals   = np.array(history.get("J_report", history["J"]))

Ftrace_vals = np.array(history.get("F_trace", history["Fh"]))
Fcol_vals   = np.array(history.get("F_col", history["Fh"]))
FTE_vals    = np.array(history.get("F_TE", Ftrace_vals))
FTM_vals    = np.array(history.get("F_TM", Ftrace_vals))

Tphys_vals  = np.array(history.get("Tphys", history["Tavg"]))
Uerr_vals   = np.array(history["Uerr"])
Pen_vals    = np.array(history["Penalty"])
Orth_vals   = np.array(history.get("col_overlap", np.full_like(Uerr_vals, np.nan)))

final_par  = history["params"][-1]
final_beta = history["beta"][-1]

fig, axes = plt.subplots(1, 4, figsize=(22, 4), tight_layout=True)

# 1. Objective values
axes[0].plot(J_vals, "k-", label="J_opt returned to NLopt")
axes[0].plot(Jrep_vals, "k--", label="J_report incl. U penalty")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Objective")
axes[0].set_title("Optimization objective")
axes[0].legend()
axes[0].grid(True)

# 2. Fidelity metrics
axes[1].plot(Ftrace_vals, "r-", label="F_trace true gate")
axes[1].plot(Fcol_vals, "b--", label="F_col avg columns")
axes[1].plot(FTE_vals, "g:", label="F_TE")
axes[1].plot(FTM_vals, "m:", label="F_TM")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Fidelity")
axes[1].set_ylim(0, 1.05)
axes[1].set_title("Hadamard fidelity metrics")
axes[1].legend()
axes[1].grid(True)

# 3. Constraint residuals
axes[2].plot(Uerr_vals, "m-", label="Uerr")
axes[2].plot(Pen_vals, "g-.", label="Fab penalty")
axes[2].plot(Orth_vals, "c:", label="|u0†u1|")
axes[2].set_xlabel("Iteration")
axes[2].set_ylabel("Residual")
axes[2].set_title("Unitarity / fabrication")
axes[2].legend()
axes[2].grid(True)

# 4. Transmission proxy
axes[3].plot(Tphys_vals, "b:", label="Tphys/Traw")
axes[3].set_xlabel("Iteration")
axes[3].set_ylabel("Transmission proxy")
if REF_NORM["TE"] is not None and REF_NORM["TM"] is not None:
    axes[3].set_title("Reference-normalized transmission")
else:
    axes[3].set_title("Raw uncalibrated output magnitude")
axes[3].legend()
axes[3].grid(True)

if mp.am_master():
    plt.savefig(f"{RUN_DIR}/convergence_metrics.png", dpi=200)
    plt.close()


# =============================================================================
# Section 16 — Verification: binarized device performance
# =============================================================================
# Re-run with beta=100 (≈ hard step function) and no penalty to measure the
# purely optical performance of the final fabrication-ready design.

if mp.am_master():
    print("=" * 60)
    print("BINARIZED VERIFICATION")
    print("=" * 60)

_, _, verify_info = evaluate_and_grad(
    params_flat    = final_par,
    beta           = 100.0,
    lambda_pen_val = 0.0,
    step_num       = "VERIFY",
)

U_final            = verify_info["U"]
fidelity_final     = verify_info["Fh"]
transmission_final = verify_info["Tavg"]
unitarity_residual = float(np.linalg.norm(U_final.conj().T @ U_final - np.eye(2), "fro"))

if mp.am_master():
    print("\nU_target =");          print(np.round(U_target, 4))
    print("\nU_meas (binarized) ="); print(np.round(U_final,  4))
    print("\n|U_target − U_meas| ="); print(np.round(np.abs(U_target - U_final), 4))
    print(f"\nFidelity  F = |Tr(U†U)|²/4  = {fidelity_final:.4f}   (ideal = 1.0)")
    print(f"Transmission T = ‖U‖²_F/2   = {transmission_final:.4f}  (lossless = 1.0)")
    print(f"Unitarity ‖U†U − I‖_F       = {unitarity_residual:.4f}  (ideal = 0.0)")
    print("=" * 60)


# =============================================================================
# Section 17 — State-by-state fidelity (no extra FDTD runs)
# =============================================================================
# Apply the measured U_final matrix analytically to six canonical input states
# and compare to U_target · ψ_in.

states = {
    "TE"    : np.array([1.0,  0.0],         dtype=complex),
    "TM"    : np.array([0.0,  1.0],         dtype=complex),
    "diag+" : np.array([1.0,  1.0],         dtype=complex) / np.sqrt(2),
    "diag-" : np.array([1.0, -1.0],         dtype=complex) / np.sqrt(2),
    "Rcirc" : np.array([1.0,  1j],          dtype=complex) / np.sqrt(2),
    "Lcirc" : np.array([1.0, -1j],          dtype=complex) / np.sqrt(2),
}

if mp.am_master():
    print(f"\n{'state':<8}  {'F_state':>8}  |ψ_meas|      |ψ_target|")
    print("-" * 52)
    for name, psi_in in states.items():
        psi_t = U_target @ psi_in
        psi_m = U_final  @ psi_in
        F = float(
            abs(np.vdot(psi_t, psi_m))**2 /
            (np.vdot(psi_t, psi_t).real * np.vdot(psi_m, psi_m).real + 1e-30)
        )
        print(f"{name:<8}  {F:>8.4f}  "
              f"[{abs(psi_m[0]):.3f}, {abs(psi_m[1]):.3f}]  "
              f"[{abs(psi_t[0]):.3f}, {abs(psi_t[1]):.3f}]")


# =============================================================================
# Section 18 — Analytical cross-check: R-D-R decomposition of Hadamard
# =============================================================================
# Hadamard = R(π/4) · D(π) · R(-π/4)  up to a global phase.
# If |Tr(U_target† U_cascade)|²/4 = 1, both unitaries are identical up to phase.

U_cascade = rotator(np.pi/4) @ retarder(np.pi) @ rotator(-np.pi/4)
phase_match = float(abs(np.trace(U_target.conj().T @ U_cascade))**2 / 4.0)

if mp.am_master():
    print(f"\nR-D-R cascade = R(-π/4)·D(π)·R(π/4) =")
    print(np.round(U_cascade, 4))
    print(f"|Tr(U_target† U_cascade)|²/4 = {phase_match:.6f}  (1.0 = same up to global phase)")


# =============================================================================
# Section 19 — Export binarized design + summary
# =============================================================================

rho_bin = (pre_process(final_par, final_beta).reshape(Nx, Ny) >= 0.5).astype(int)

if mp.am_master():
    np.savetxt(final_design_csv, rho_bin, fmt="%d", delimiter=",")

fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)
ax.imshow(rho_bin.T, extent=[-dr_lx/2, dr_lx/2, -dr_ly/2, dr_ly/2],
          origin="lower", cmap="binary", vmin=0, vmax=1)
ax.set_title("Final binarized design  (black = Si, white = air)")
ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")

if mp.am_master():
    plt.savefig(f"{RUN_DIR}/final_design.png", dpi=200)
    plt.close()
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Runtime              : {datetime.now() - START_TIME}")
    print(f"Target unitary       : Hadamard")
    print(f"Wavelength           : {wl} µm")
    print(f"Design region        : {dr_lx} µm × {dr_ly} µm  ({Nx}×{Ny} pixels)")
    print(f"Resolution           : {global_res} px/µm")
    print(f"Beta                 : {beta_min} → {beta_max}  (ramp start = iter {beta_ramp_start})")
    print(f"λ_pen                : 0 → {lambda_pen_max}  (ramp start = 40% of opt_steps)")
    print(f"Iterations           : {len(history['J'])}")
    print(f"Final J              : {history['J'][-1]:.4f}")
    print(f"Final fidelity F     : {history['Fh'][-1]:.4f}  (ideal = 1.0)")
    print(f"Binarized fidelity F : {fidelity_final:.4f}  (ideal = 1.0)")
    print(f"Binarized T          : {transmission_final:.4f}  (lossless = 1.0)")
    print(f"Unitarity residual   : {unitarity_residual:.4f}  (ideal = 0.0)")
    print(f"Design CSV           : {final_design_csv}")
    print("=" * 60)
    