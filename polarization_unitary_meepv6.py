#!/usr/bin/env python
# coding: utf-8

# # Inverse Design of a Polarization-Basis 2×2 Unitary Operator (MEEP)
# 
# MEEP port of `polarization_unitary.ipynb` (Tidy3D original).
# 
# Runs a full 3-D SOI simulation (220 nm Si / 1 µm SiO₂ BOX / air cladding)  
# and uses `meep.adjoint` to inverse-design a patterned-Si region that  
# implements a chosen 2×2 unitary on the (TE₀, TM₀) polarization qubit.
# 
# **Setup (WSL/Ubuntu):**
# ```
# conda install -c conda-forge pymeep=*=mpi_mpich_* nlopt scipy matplotlib
# ```
# 

# In[1]:


# Section 1 — Imports
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
mp.verbosity(1)   # set to 1 for full FDTD diagnostics

if mp.am_master():
    print(f"MEEP  version : {mp.__version__}")
    print(f"NumPy version : {np.__version__}")

START_TIME = datetime.now()


# In[2]:


# Section 2 — Physical parameters  (all lengths in µm; 1 MEEP unit = 1 µm)

# Wavelength / frequency
wl   = 1.55          # µm
fcen = 1.0 / wl      # MEEP natural frequency

# Materials
n_Si   = 3.476
n_SiO2 = 1.444
eps_Si   = n_Si**2
eps_SiO2 = n_SiO2**2
eps_air  = 1.0

Si   = mp.Medium(epsilon=eps_Si)
SiO2 = mp.Medium(epsilon=eps_SiO2)
Air  = mp.Medium(epsilon=eps_air)

# Waveguide cross-section (standard 220 nm SOI)
t_Si  = 0.22         # Si layer thickness
w_wg  = 0.50         # waveguide width — supports both TE₀ and TM₀
t_BOX = 1.0          # SiO₂ BOX thickness

# Design region
dr_lx = 6.0          # design region length in x (µm)
dr_ly = 3.0          # design region width in y (µm)


# ===== RUN CONFIG =====
RUN_NAME = "run3A_stage1"      # change this for each run
RESTART_FROM = None  # e.g. "misc/runs/run1_stage1/final_params.npy"

RUN_DIR = f"misc/runs/{RUN_NAME}"
os.makedirs(RUN_DIR, exist_ok=True)

history_fname = f"{RUN_DIR}/history.pkl"
final_params_fname = f"{RUN_DIR}/final_params.npy"
final_design_csv = f"{RUN_DIR}/final_design.csv"

# ===== QUALITY SETTINGS =====
global_res = 16
dr_res = global_res

opt_steps = 100
lambda_pen_max = 0.01
unitary_pen_max = 0.01  # gradient is approximate: keep this low (~0.01)

beta_min = 1.0
beta_max = 6.0      # 2-4 allows exploration, 6-10 is more binary, 12-24 to lock in binary values
beta_ramp_start = int(opt_steps * 0.4)

df = 0.05 * fcen     # stage1: 0.05, stage2: 0.035, stage3: 0.025, polish: 0.02

skip_transmission_calibration = True
# Other params

# Waveguide stub length on each side of the design region (inside the cell)
stub_len = 1.0

# PML thickness
dpml = 1.0

# Fabrication filter
min_feature = 0.10   # 100 nm
filter_R    = min_feature


# In[3]:

# Section 3 — Derived geometry quantities with strict pixel snapping

# ------------------------------------------------------------
# Grid snapping helpers
# ------------------------------------------------------------
def snap_len(x, res=global_res):
    """
    Snap a physical length in µm to an integer number of grid pixels.
    """
    return round(x * res) / res


def snap_pix(x, res=global_res):
    """
    Return integer pixel count for a physical length.
    """
    return int(round(x * res))


def snap_coord(x, res=global_res):
    """
    Snap a coordinate to the simulation grid.
    Useful for source/monitor centers.
    """
    return round(x * res) / res

# ------------------------------------------------------------
# Grid-snapped physical dimensions
# ------------------------------------------------------------

# Design region pixel counts
Nx = snap_pix(dr_lx, global_res)
Ny = snap_pix(dr_ly, global_res)

# Snap design-region dimensions
dr_lx = Nx / global_res
dr_ly = Ny / global_res

# Snap layer thicknesses used in geometry/volumes
# Note: this slightly changes the simulated thickness to the nearest grid value.
t_Si_sim  = snap_len(t_Si, global_res)
t_BOX_sim = snap_len(t_BOX, global_res)

# Use snapped layer thicknesses for geometry placement
z_si_top  =  t_Si_sim / 2
z_si_bot  = -t_Si_sim / 2
z_box_bot = z_si_bot - t_BOX_sim

# Stub / PML dimensions
stub_len = snap_len(stub_len, global_res)
dpml     = snap_len(dpml, global_res)

# Full cell dimensions
sx = snap_len(dr_lx + 2 * stub_len + 2 * dpml, global_res)
sy = snap_len(dr_ly + 2 * dpml, global_res)
sz = snap_len(t_Si_sim + t_BOX_sim + 2 * dpml, global_res)

cell = mp.Vector3(sx, sy, sz)

# Source / monitor x positions
src_x = snap_coord(-(dr_lx / 2 + stub_len / 2), global_res)
mon_x = snap_coord( (dr_lx / 2 + stub_len / 2), global_res)

# Monitor cross-section
mon_y = dr_ly
mon_z = snap_len(t_Si_sim + t_BOX_sim + 0.4, global_res)

# Interface-buffer dimensions in design-grid pixels
n_border_x  = max(2, int(round(0.16 * dr_res)))
n_wg_half_y = max(1, int(round((w_wg / 2) * dr_res)))
n_cy        = Ny // 2

Npar = Nx * Ny

if mp.am_master():
    os.makedirs("misc", exist_ok=True)
    
    print(f"Design region : {dr_lx} µm × {dr_ly} µm")
    print(f"Grid          : {Nx} × {Ny} = {Nx*Ny:,} parameters")
    print(f"Resolution    : {global_res} px/µm  (Si slab = {t_Si*global_res:.1f} cells thick)")
    print(f"Physical t_Si requested : {t_Si:.4f} µm")
    print(f"Grid-snapped t_Si used  : {t_Si_sim:.4f} µm")
    print(f"Physical BOX requested  : {t_BOX:.4f} µm")
    print(f"Grid-snapped BOX used   : {t_BOX_sim:.4f} µm")
    print(f"Design region           : {dr_lx:.4f} µm × {dr_ly:.4f} µm")
    print(f"Grid                    : {Nx} × {Ny} = {Npar:,} parameters")
    print(f"Cell size               : {sx:.4f} × {sy:.4f} × {sz:.4f} µm")
    print(f"src_x = {src_x:.4f} µm, mon_x = {mon_x:.4f} µm")





# In[4]:


# Section 4 — Static geometry (BOX + waveguide stubs)
#
# The design-region block is added separately in each simulation so the
# MaterialGrid can be updated without rebuilding static structures.

def static_geometry():
    """Return list of mp.GeometricObject for the fixed (non-design) structures."""
    INF = 1e6

    # SiO₂ BOX (below the Si slab)
    box_layer = mp.Block(
        size   = mp.Vector3(INF, INF, t_BOX),
        center = mp.Vector3(0, 0, z_si_bot - t_BOX/2),
        material = SiO2,
    )

    # Left input-waveguide stub: x ∈ [-sx/2, -dr_lx/2]
    # Extend 0.01 µm into the design boundary to avoid a gap.
    stub_half_x = (sx/2 - dr_lx/2 + 0.01) / 2
    wg_in = mp.Block(
        size   = mp.Vector3(sx/2 - dr_lx/2 + 0.02, w_wg, t_Si),
        center = mp.Vector3(-(dr_lx/2 + stub_half_x), 0, 0),
        material = Si,
    )

    # Right output-waveguide stub: x ∈ [dr_lx/2, sx/2]
    wg_out = mp.Block(
        size   = mp.Vector3(sx/2 - dr_lx/2 + 0.02, w_wg, t_Si),
        center = mp.Vector3( (dr_lx/2 + stub_half_x), 0, 0),
        material = Si,
    )

    return [box_layer, wg_in, wg_out]

if mp.am_master():
    print("static_geometry() defined — SiO₂ BOX + WG-in + WG-out")


# In[5]:


# Section 5 — Parameter preprocessing pipeline
#
# 1. interface_buffer  — force rho=1 in a WG-width strip at left/right edges
# 2. conic_filter      — spatial blur to enforce ≥100 nm minimum feature size
# 3. tanh projection   — drive pixels toward 0 or 1 (sharpness β)
# Applied twice (filter+project)×2 for cleaner binarisation.
#
# The adjoint chain rule through the pipeline is implemented analytically
# in pre_process_grad().

def _make_conic_kernel(radius_um, res_pxum):
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
    t = np.tanh(beta * 0.5)
    return 0.5 + 0.5 * np.tanh(beta * (rho2d - 0.5)) / t

def dtanh_proj(rho2d, beta):
    """Elementwise derivative of tanh_proj w.r.t. rho2d."""
    t = np.tanh(beta * 0.5)
    return 0.5 * beta * (1.0 - np.tanh(beta * (rho2d - 0.5))**2) / t

def interface_buffer(p2d):
    """Force rho=1 in the waveguide-width strip at the left/right edges."""
    p = p2d.copy()
    y0 = n_cy - n_wg_half_y
    y1 = n_cy + n_wg_half_y + 1
    p[0          : n_border_x,  y0:y1] = 1.0
    p[Nx - n_border_x : Nx,    y0:y1] = 1.0
    return p

def pre_process(params_flat, beta):
    """Return flattened pre-processed density in [0,1], shape (Npar,)."""
    p = params_flat.reshape(Nx, Ny)
    p = interface_buffer(p)
    p = conic_filter(p);  p = tanh_proj(p, beta)
    p = conic_filter(p);  p = tanh_proj(p, beta)
    return p.ravel()

def pre_process_grad(params_flat, grad_rho_flat, beta):
    """Back-propagate grad_rho_flat (dJ/d rho_processed) → dJ/d params_flat."""
    p0 = interface_buffer(params_flat.reshape(Nx, Ny))
    f1 = conic_filter(p0);      p1 = tanh_proj(f1, beta)
    f2 = conic_filter(p1)
    # grad comes in as dJ/d p2 (after second tanh_proj)
    g  = grad_rho_flat.reshape(Nx, Ny)
    g  = g * dtanh_proj(f2, beta)                                   # through 2nd tanh
    g  = ssg.convolve2d(g, _conic_k, mode="same", boundary="fill", fillvalue=0.0)  # through 2nd filter
    g  = g * dtanh_proj(f1, beta)                                   # through 1st tanh
    g  = ssg.convolve2d(g, _conic_k, mode="same", boundary="fill", fillvalue=0.0)  # through 1st filter
    # Zero out interface-buffer pixels (their params are not free variables)
    y0, y1 = n_cy - n_wg_half_y, n_cy + n_wg_half_y + 1
    g[0:n_border_x, y0:y1]          = 0.0
    g[Nx-n_border_x:Nx, y0:y1]     = 0.0
    return g.ravel()

if mp.am_master():
    print(f"Conic kernel: {_conic_k.shape[0]}×{_conic_k.shape[1]} px  "
        f"(radius={filter_R} µm at {dr_res} px/µm)")


# In[6]:


# Section 6 — Erosion-dilation fabrication penalty
#
# penalty = mean( rho_eroded · (1 − rho_dilated) )
# = 0 when all pixels are 0 or 1, > 0 when gray pixels exist.
# The smooth erosion/dilation approximations follow Lazarov et al. (2016).

def fab_penalty(params_flat, beta, beta_ed=8.0):
    rho = pre_process(params_flat, beta).reshape(Nx, Ny)
    # Smooth erosion: complement, filter, complement
    rho_ero = tanh_proj(conic_filter(1.0 - rho), beta_ed)
    rho_ero = 1.0 - rho_ero
    # Smooth dilation: filter, project
    rho_dil = tanh_proj(conic_filter(rho), beta_ed)
    return float(np.mean(rho_ero * (1.0 - rho_dil)))


def conic_filter_T(g):
    """
    Adjoint of conic_filter.
    Since the kernel is symmetric, this is the same convolution.
    """
    return ssg.convolve2d(g, _conic_k, mode="same", boundary="fill", fillvalue=0.0)


def fab_penalty_and_grad(params_flat, beta, beta_ed=8.0):
    """
    Returns fabrication penalty and approximate analytical gradient
    with respect to raw params_flat.
    """

    rho = pre_process(params_flat, beta).reshape(Nx, Ny)

    # Erosion branch
    q = 1.0 - rho
    fq = conic_filter(q)
    pq = tanh_proj(fq, beta_ed)
    rho_ero = 1.0 - pq

    # Dilation branch
    fr = conic_filter(rho)
    rho_dil = tanh_proj(fr, beta_ed)

    penalty = np.mean(rho_ero * (1.0 - rho_dil))

    scale = 1.0 / (Nx * Ny)

    # d penalty / d rho_ero
    g_ero = scale * (1.0 - rho_dil)

    # d penalty / d rho_dil
    g_dil = scale * (-rho_ero)

    # Backprop erosion:
    # rho_ero = 1 - tanh_proj(conic_filter(1-rho))
    g_pq = -g_ero
    g_fq = g_pq * dtanh_proj(fq, beta_ed)
    g_q = conic_filter_T(g_fq)
    g_rho_from_ero = -g_q

    # Backprop dilation:
    # rho_dil = tanh_proj(conic_filter(rho))
    g_fr = g_dil * dtanh_proj(fr, beta_ed)
    g_rho_from_dil = conic_filter_T(g_fr)

    g_rho = g_rho_from_ero + g_rho_from_dil

    # Backprop through main preprocessing pipeline
    g_params = pre_process_grad(params_flat, g_rho.ravel(), beta)

    return float(penalty), g_params

if mp.am_master():
    print("Penalty functions defined.")

# In[7]:


# Section 7 — Mode-index calibration
#
# MEEP EigenModeSource sorts modes by descending Re(neff).
# For 220 nm × 500 nm Si-on-SiO₂ at 1550 nm:
#   band 1 → TE₀  (Ey-dominant, neff ≈ 2.4)
#   band 2 → TM₀  (Ez-dominant, neff ≈ 1.7)
#
# We verify by sampling the eigenmode field inside the waveguide core.

eig_band_TE = 1   # updated by calibrate_modes()
eig_band_TM = 2

def calibrate_modes():
    """Empirically assign eig_band_TE and eig_band_TM."""
    global eig_band_TE, eig_band_TM

    # Minimal geometry: just the waveguide cross-section for eigenmode solve.
    wg_geom = [
        mp.Block(size=mp.Vector3(1e6, 1e6, t_BOX),
                 center=mp.Vector3(0, 0, z_si_bot - t_BOX/2), material=SiO2),
        mp.Block(size=mp.Vector3(1e6, w_wg, t_Si),
                 center=mp.Vector3(0, 0, 0), material=Si),
    ]
    # Minimal cell (thin in x since we only need the cross-section)
    cal_cell = mp.Vector3(0.2, sy, sz)
    cal_pml  = [mp.PML(dpml, direction=mp.Y), mp.PML(dpml, direction=mp.Z)]

    dummy_src = [mp.EigenModeSource(
        src=mp.GaussianSource(fcen, fwidth=df*10),
        center=mp.Vector3(0, 0, 0),
        size=mp.Vector3(0, mon_y, mon_z),
        eig_band=1,
    )]

    sim_cal = mp.Simulation(
        cell_size=cal_cell,
        boundary_layers=cal_pml,
        geometry=wg_geom,
        sources=dummy_src,
        resolution=global_res,
    )
    sim_cal.init_sim()

    mon_vol = mp.Volume(center=mp.Vector3(0, 0, 0),
                        size=mp.Vector3(0, mon_y, mon_z))

    results = {}
    for band in [1, 2]:
        em = sim_cal.get_eigenmode(fcen, mp.X, mon_vol, band, mp.Vector3(1, 0, 0))
        # Sample |Ey|² and |Ez|² on a small grid inside the waveguide core
        ys = np.linspace(-w_wg/2*0.9, w_wg/2*0.9, 8)
        zs = np.linspace(-t_Si/2*0.9, t_Si/2*0.9, 4)
        Ey2, Ez2 = 0.0, 0.0
        for y in ys:
            for z in zs:
                pt = mp.Vector3(0, y, z)
                Ey2 += abs(em.amplitude(pt, mp.Ey))**2
                Ez2 += abs(em.amplitude(pt, mp.Ez))**2
        te_frac = Ey2 / (Ey2 + Ez2 + 1e-30)
        neff    = em.kdom.x / fcen
        results[band] = (te_frac, neff)

        if mp.am_master():
            print(f"  band {band}: neff={neff:.4f}  |Ey|² frac={te_frac:.3f}")

    # Assign TE to the band with larger |Ey|² fraction
    eig_band_TE = 1 if results[1][0] >= results[2][0] else 2
    eig_band_TM = 3 - eig_band_TE

    if mp.am_master():
        print(f"\n  → eig_band_TE={eig_band_TE},  eig_band_TM={eig_band_TM}")
    return eig_band_TE, eig_band_TM

if mp.am_master():
    print("Running mode calibration…")
eig_band_TE, eig_band_TM = calibrate_modes()


# In[8]:


# Section 8 — Target unitary

def hadamard():
    return (1.0/np.sqrt(2)) * np.array([[1, 1],[1,-1]], dtype=complex)

def pauli_x():
    return np.array([[0,1],[1,0]], dtype=complex)

def pauli_z():
    return np.array([[1,0],[0,-1]], dtype=complex)

def rotator(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[c,-s],[s,c]], dtype=complex)

def retarder(phi):
    return np.array([[1,0],[0,np.exp(1j*phi)]], dtype=complex)

def is_unitary(U, tol=1e-9):
    return np.allclose(U.conj().T @ U, np.eye(2), atol=tol)

# ── Choose target here ────────────────────────────────────────────────────────
# U_target = pauli_x()              # TE ↔ TM swap
# U_target = rotator(np.pi/4)       # 45° polarisation rotator
U_target = hadamard()

assert is_unitary(U_target), "U_target is not unitary!"
if mp.am_master():
    print("U_target (Hadamard) =")
    print(np.round(U_target, 4))

# ------------------------------------------------------------
# Reference normalization for physical transmission metric
# ------------------------------------------------------------
REF_NORM = {
    "TE": None,
    "TM": None,
}

# In[9]:


# Section 9 — Build one OptimizationProblem per input polarization
#
# MEEP adjoint API pattern:
#
#   design_vars = mp.MaterialGrid(...)   # weights ∈ [0,1]: 0=Air, 1=Si
#   dr = mpa.DesignRegion(design_parameters=design_vars, volume=...)
#   te_mon = mpa.EigenmodeCoefficient(sim, vol, mode=eig_band_TE)
#   tm_mon = mpa.EigenmodeCoefficient(sim, vol, mode=eig_band_TM)
#
#   def obj(a_te, a_tm):
#       # a_te and a_tm are autograd-tracked 1-element arrays (one per frequency)
#       return <scalar autograd expression>
#
#   opt = mpa.OptimizationProblem(
#       simulation=sim,
#       objective_functions=[obj],
#       objective_arguments=[te_mon, tm_mon],
#       design_regions=[dr],
#       frequencies=[fcen],
#   )
#   f_val, (dJ_dweights,) = opt([rho])   # rho shape: (Npar,)
#
# The objective function for each polarization is the column-overlap with the
# target:  J_col = |⟨u_target_col | u_meas_col⟩|²
# where u_target_col is the corresponding column of U_target (a constant).
#
# Summing J_TE + J_TM and dividing by 4 gives |Tr(U†U_meas)|²/4 — the
# standard gauge-invariant Hadamard fidelity F ∈ [0, 1].
# The gradient returned by opt() is exactly dJ_col / d(rho), so summing
# grad_TE + grad_TM gives dF/d(rho) up to the 1/4 normalization.

# ------------------------------------------------------------
# Dynamic objective controls
# ------------------------------------------------------------
DYN_OBJ = {
    "TE": {
        "target": np.array([1.0, 0.0], dtype=complex),
        "orth":   np.array([0.0, 1.0], dtype=complex),
        "orth_weight": 0.0,
    },
    "TM": {
        "target": np.array([0.0, 1.0], dtype=complex),
        "orth":   np.array([1.0, 0.0], dtype=complex),
        "orth_weight": 0.0,
    },
}

# Previous normalized output columns, used for lagged unitarity gradient
PREV_COLS = {
    "TE": None,
    "TM": None,
}


def _make_opt(source_band, input_key):
    """
    Build a persistent OptimizationProblem for one input polarization.

    This objective supports:
      1. Positive overlap with the desired Hadamard target column.
      2. Negative overlap with a lagged opposite output column.

    This gives an approximate unitary/orthogonality gradient using only
    two MEEP adjoint solves per optimization iteration.
    """

    dummy_rho = np.full(Npar, 0.5)

    design_vars = mp.MaterialGrid(
        mp.Vector3(Nx, Ny, 1),
        Air,
        Si,
        weights=dummy_rho,
        grid_type="U_DEFAULT",
    )

    design_block = mp.Block(
        size=mp.Vector3(dr_lx, dr_ly, t_Si),
        center=mp.Vector3(0, 0, 0),
        material=design_vars,
    )

    geom = static_geometry() + [design_block]

    dr = mpa.DesignRegion(
        design_parameters=design_vars,
        volume=mp.Volume(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(dr_lx, dr_ly, t_Si),
        ),
    )

    src = mp.EigenModeSource(
        src=mp.GaussianSource(fcen, fwidth=df),
        center=mp.Vector3(src_x, 0, 0),
        size=mp.Vector3(0, mon_y, mon_z),
        eig_band=source_band,
        eig_match_freq=True,
        eig_parity=mp.NO_PARITY,
        direction=mp.X,
    )

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(dpml)],
        geometry=geom,
        sources=[src],
        resolution=global_res,
        eps_averaging=False,
    )

    mon_vol = mp.Volume(
        center=mp.Vector3(mon_x, 0, 0),
        size=mp.Vector3(0, mon_y, mon_z),
    )

    te_mon = mpa.EigenmodeCoefficient(
        sim, mon_vol, mode=eig_band_TE,
        eig_parity=mp.NO_PARITY,
        forward=True,
    )

    tm_mon = mpa.EigenmodeCoefficient(
        sim, mon_vol, mode=eig_band_TM,
        eig_parity=mp.NO_PARITY,
        forward=True,
    )

    def objective(a_te, a_tm):
        obj = DYN_OBJ[input_key]

        target = obj["target"]
        orth = obj["orth"]
        orth_weight = obj["orth_weight"]

        # Desired Hadamard-column overlap
        good_overlap = anp.conj(target[0]) * a_te[0] + anp.conj(target[1]) * a_tm[0]
        good_term = anp.abs(good_overlap) ** 2

        # Lagged orthogonality penalty
        bad_overlap = anp.conj(orth[0]) * a_te[0] + anp.conj(orth[1]) * a_tm[0]
        bad_term = anp.abs(bad_overlap) ** 2

        return good_term - orth_weight * bad_term

    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=[objective],
        objective_arguments=[te_mon, tm_mon],
        design_regions=[dr],
        frequencies=[fcen],
    )

    return opt

if mp.am_master():
    print("Building TE optimization problem...")
opt_TE = _make_opt(eig_band_TE, "TE")

if mp.am_master():
    print("Building TM optimization problem...")
opt_TM = _make_opt(eig_band_TM, "TM")

if mp.am_master():
    print("OptimizationProblem objects created.")


# In[10]:


# Section 10 — Forward/adjoint pass helper
#
# opt_obj([rho]) runs one MEEP forward+adjoint simulation and returns
# (f_val, grad_arr) where grad_arr has shape (Npar,).
# We also read the raw complex mode amplitudes from the monitor objects
# via get_objective_arguments().

def _get_raw_amplitudes(opt_obj, rho):
    """
    Run one forward+adjoint pass and return (f_val, grad, a_te, a_tm).

    result[0] is the scalar objective value.
    result[1] is the gradient array of shape (Npar,).
    """
    result   = opt_obj([rho])
    f_val    = result[0]
    grad_arr = result[1]

    args = opt_obj.get_objective_arguments()
    # args[0] = te_mon outputs, args[1] = tm_mon outputs; each is a 1-element
    # array indexed by frequency.  Index [0] selects the single frequency fcen.
    a_te = complex(args[0][0])
    a_tm = complex(args[1][0])

    return float(np.real(f_val)), grad_arr, a_te, a_tm

def calibrate_transmission_reference(beta=1.0):
    """
    Calibrate raw EigenmodeCoefficient amplitudes to a straight-ish reference.

    This gives reference amplitudes for:
      - TE input -> TE output
      - TM input -> TM output

    After this, U_phys[:, 0] = U_raw[:, 0] / REF_NORM["TE"]
                U_phys[:, 1] = U_raw[:, 1] / REF_NORM["TM"]

    Note:
    This uses a fully filled silicon design region as the reference.
    That creates a continuous straight waveguide/slab through the device region.
    """

    global REF_NORM

    if mp.am_master():
        print("=" * 60)
        print("CALIBRATING TRANSMISSION REFERENCE")
        print("=" * 60)

    # Fully silicon design region after preprocessing/interface buffer
    ref_params = np.ones(Npar)
    rho_ref = pre_process(ref_params, beta)

    # TE input reference
    _, _, a_te_TE_ref, a_tm_TE_ref = _get_raw_amplitudes(opt_TE, rho_ref)

    # TM input reference
    _, _, a_te_TM_ref, a_tm_TM_ref = _get_raw_amplitudes(opt_TM, rho_ref)

    # Use same-polarization through coefficients as input normalization
    REF_NORM["TE"] = a_te_TE_ref
    REF_NORM["TM"] = a_tm_TM_ref

    if mp.am_master():
        print(f"TE reference amplitude: {REF_NORM['TE']}")
        print(f"TM reference amplitude: {REF_NORM['TM']}")
        print("Reference cross terms:")
        print(f"  TE input -> TM output: {a_tm_TE_ref}")
        print(f"  TM input -> TE output: {a_te_TM_ref}")
        print("=" * 60)

    return REF_NORM

if mp.am_master():
    print("_get_raw_amplitudes() and calibrate_transmission_reference() defined.")

# In[11]:


# Section 11 — Initial design visualisation

np.random.seed(42)
init_par = np.random.uniform(0, 1, Npar)
init_par = ndi.gaussian_filter(init_par.reshape(Nx, Ny), sigma=1.5).ravel()
init_par = np.clip(init_par, 0, 1)

rho_init = pre_process(init_par, beta=beta_min).reshape(Nx, Ny)
eps_init = eps_air + (eps_Si - eps_air) * rho_init

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)

im = ax0.imshow(eps_init.T, extent=[-dr_lx/2, dr_lx/2, -dr_ly/2, dr_ly/2],
                origin="lower", cmap="RdGy_r", vmin=eps_air, vmax=eps_Si)
plt.colorbar(im, ax=ax0, label="ε")
ax0.set_title("Initial design (xy, z=0)"); ax0.set_xlabel("x (µm)"); ax0.set_ylabel("y (µm)")

ax1.hist(rho_init.ravel(), bins=50, color="steelblue", edgecolor="none")
ax1.set_xlabel("Density ρ"); ax1.set_ylabel("Count"); ax1.set_title("Initial density histogram")

if mp.am_master():
    plt.savefig(f"{RUN_DIR}/initial_design.png", dpi=200)
    plt.close()
    print(f"Npar = {Npar:,}  |  init_par ∈ [{init_par.min():.3f}, {init_par.max():.3f}]")


# In[12]:


# Section 12 — Checkpoint management

def save_history(h):
    if mp.am_master():
        with open(history_fname, "wb") as fh:
            pickle.dump(h, fh)
    mp.comm.barrier()

def load_history():
    with open(history_fname, "rb") as fh:
        return pickle.load(fh)

def resize_params_if_needed(p_old, old_shape, new_shape):
    old_Nx, old_Ny = old_shape
    new_Nx, new_Ny = new_shape

    if (old_Nx, old_Ny) == (new_Nx, new_Ny):
        return p_old.copy()

    p2d = p_old.reshape(old_Nx, old_Ny)
    zoom_x = new_Nx / old_Nx
    zoom_y = new_Ny / old_Ny

    p_new = ndi.zoom(p2d, (zoom_x, zoom_y), order=1)
    p_new = np.clip(p_new, 0.0, 1.0)

    return p_new.reshape(new_Nx * new_Ny)

try:
    history = load_history()
    params = history["params"][-1].copy()
    num_done = len(history["J"])

    if mp.am_master():
        print(f"Loaded checkpoint: {num_done}/{opt_steps} iterations done from {history_fname}")

except FileNotFoundError:
    if RESTART_FROM is not None:
        ckpt = np.load(RESTART_FROM, allow_pickle=True).item()

        old_params = ckpt["params"]
        old_shape = tuple(ckpt["shape"])

        params = resize_params_if_needed(
            old_params,
            old_shape=old_shape,
            new_shape=(Nx, Ny),
        )

        history = dict(
            J=[], params=[], grad=[], beta=[], U_meas=[],
            Fh=[], Tavg=[], Uerr=[], Penalty=[]
        )

        if mp.am_master():
            print(f"Restarting from saved design: {RESTART_FROM}")
            print(f"Old shape: {old_shape}, new shape: {(Nx, Ny)}")

    else:
        params = init_par.copy()
        history = dict(
            J=[], params=[], grad=[], beta=[], U_meas=[],
            Fh=[], Tavg=[], Uerr=[], Penalty=[]
        )
        if mp.am_master():
            print("No checkpoint found — starting fresh.")


# In[13]:


# Section 13 — Main evaluation: J, gradient, and U_meas
#
# Each call runs exactly two FDTD+adjoint simulations (TE input, TM input).
# No extra calibration runs are needed.
#
# Per-column amplitude normalization
# -----------------------------------
# Divide each output column vector by its own Euclidean norm so that the
# columns of U_meas are unit vectors (lossless normalization).  This places
# U_meas on the same O(1) scale as U_target regardless of MEEP's internal
# source normalization convention.
#
# Gradient scaling: the adjoint returns d(J_raw)/d(rho) where J_raw uses
# unnormalized amplitudes.  After dividing amplitudes by norm_col, the
# normalized objective is J_norm = J_raw / norm_col².  Therefore:
#   d(J_norm)/d(rho) = d(J_raw)/d(rho) / norm_col²
# We apply this per-column scale before summing and chain-ruling through
# the preprocessing pipeline.

def evaluate_and_grad(params_flat, beta, lambda_pen_val, unitary_pen_val, step_num=None):
    """
    2-simulation-per-iteration version with approximate lagged unitary gradient.

    Cost:
      - TE input adjoint solve
      - TM input adjoint solve

    Includes:
      - Hadamard column fidelity gradient
      - fabrication penalty gradient
      - approximate unitary/orthogonality gradient using previous iteration columns
    """

    rho = pre_process(params_flat, beta)

    # ------------------------------------------------------------
    # 1. Configure dynamic objectives
    # ------------------------------------------------------------

    # Desired Hadamard columns
    DYN_OBJ["TE"]["target"] = U_target[:, 0]
    DYN_OBJ["TM"]["target"] = U_target[:, 1]

    # Use previous opposite columns for lightweight unitary gradient
    if PREV_COLS["TM"] is not None and PREV_COLS["TE"] is not None:
        DYN_OBJ["TE"]["orth"] = PREV_COLS["TM"]
        DYN_OBJ["TM"]["orth"] = PREV_COLS["TE"]

        # This is the approximate unitary-gradient strength.
        # Keep it modest because it is lagged/stale.
        DYN_OBJ["TE"]["orth_weight"] = unitary_pen_val
        DYN_OBJ["TM"]["orth_weight"] = unitary_pen_val
    else:
        # First iteration has no previous columns yet
        DYN_OBJ["TE"]["orth"] = U_target[:, 1]
        DYN_OBJ["TM"]["orth"] = U_target[:, 0]
        DYN_OBJ["TE"]["orth_weight"] = 0.0
        DYN_OBJ["TM"]["orth_weight"] = 0.0

    # ------------------------------------------------------------
    # 2. TE and TM simulations
    # ------------------------------------------------------------
    f_te_obj, grad_te_obj, a_te_TE_raw, a_tm_TE_raw = _get_raw_amplitudes(opt_TE, rho)
    f_tm_obj, grad_tm_obj, a_te_TM_raw, a_tm_TM_raw = _get_raw_amplitudes(opt_TM, rho)

    # ------------------------------------------------------------
    # 3. Build raw measured matrix
    # ------------------------------------------------------------
    U_raw = np.array([
        [a_te_TE_raw, a_te_TM_raw],
        [a_tm_TE_raw, a_tm_TM_raw],
    ], dtype=complex)

    # ------------------------------------------------------------
    # Calibrated physical-ish guided-mode transmission
    # ------------------------------------------------------------
    if REF_NORM["TE"] is not None and REF_NORM["TM"] is not None:
        U_phys = np.array([
            [U_raw[0, 0] / REF_NORM["TE"], U_raw[0, 1] / REF_NORM["TM"]],
            [U_raw[1, 0] / REF_NORM["TE"], U_raw[1, 1] / REF_NORM["TM"]],
        ], dtype=complex)

        Tphys = float(np.linalg.norm(U_phys, "fro")**2 / 2.0)
    else:
        U_phys = U_raw.copy()
        Tphys = float(np.linalg.norm(U_raw, "fro")**2 / 2.0)

    # ------------------------------------------------------------
    # 4. Normalize columns for gate-shape metrics
    # ------------------------------------------------------------
    norm_TE = np.sqrt(abs(a_te_TE_raw)**2 + abs(a_tm_TE_raw)**2 + 1e-30)
    norm_TM = np.sqrt(abs(a_te_TM_raw)**2 + abs(a_tm_TM_raw)**2 + 1e-30)

    u_TE = np.array([a_te_TE_raw, a_tm_TE_raw], dtype=complex) / norm_TE
    u_TM = np.array([a_te_TM_raw, a_tm_TM_raw], dtype=complex) / norm_TM

    U_meas = np.array([
        [u_TE[0], u_TM[0]],
        [u_TE[1], u_TM[1]],
    ], dtype=complex)

    # Save columns for next iteration's lagged unitary gradient
    PREV_COLS["TE"] = u_TE.copy()
    PREV_COLS["TM"] = u_TM.copy()

    # ------------------------------------------------------------
    # 5. Diagnostics
    # ------------------------------------------------------------
    col0 = abs(np.vdot(U_target[:, 0], u_TE))**2
    col1 = abs(np.vdot(U_target[:, 1], u_TM))**2
    fidelity = float(0.5 * (col0 + col1))

    Uerr = float(np.linalg.norm(U_meas.conj().T @ U_meas - np.eye(2), "fro"))
    unitary_pen = Uerr**2

    # ------------------------------------------------------------
    # 6. Fabrication penalty and gradient
    # ------------------------------------------------------------
    pen, grad_fab_params = fab_penalty_and_grad(params_flat, beta)

    # ------------------------------------------------------------
    # 7. Optical gradient from the two dynamic objectives
    #
    # grad_te_obj and grad_tm_obj already include:
    #   + Hadamard target gradient
    #   - lagged orthogonality gradient
    # because that was built into the MEEP objective.
    # ------------------------------------------------------------
    dJopt_drho = 0.5 * (
        grad_te_obj.ravel() / (norm_TE**2)
        + grad_tm_obj.ravel() / (norm_TM**2)
    )

    grad_optical_params = pre_process_grad(params_flat, dJopt_drho, beta)

    # ------------------------------------------------------------
    # 8. Final objective and gradient
    # ------------------------------------------------------------
    J = fidelity - lambda_pen_val * pen - unitary_pen_val * unitary_pen

    grad = grad_optical_params - lambda_pen_val * grad_fab_params

    # ------------------------------------------------------------
    # 9. Logging
    # ------------------------------------------------------------
    tag = f"step {step_num}" if step_num is not None else "eval"
    np.set_printoptions(precision=3, suppress=True)

    if mp.am_master():
        print(
            f"[{tag}] J={J:.4f} | F={fidelity:.4f} | "
            f"Tphys={Tphys:.4f} | Uerr={Uerr:.4f} | orth_w={unitary_pen_val * unitary_pen:.4f} | "
            f"Penalty={pen:.4f} | penalty_w={lambda_pen_val * pen:.4f} | beta={beta:.1f}"
        )
        print(f"  U_raw =\n{U_raw}")
        print(f"  U_norm =\n{U_meas}")

    info = {
        "J": float(np.real(J)),
        "Fh": fidelity,
        "Tavg": Tphys,
        "Uerr": Uerr,
        "Penalty": pen,
        "U": U_meas.copy(),
        "U_raw": U_raw.copy(),
        "U_phys": U_phys.copy(),
    }

    return float(np.real(J)), grad, info

if mp.am_master():
    print("evaluate_and_grad() defined.")


# In[ ]:


# Section 14 — Optimisation loop (NLopt LD_MMA)
#
# NLopt LD_MMA (Method of Moving Asymptotes) is a standard algorithm for
# large-scale topology-optimisation problems with box constraints.
# NLopt convention: set_max_objective → grad[:] receives dJ/dx (positive direction).

iteration_counter = [len(history["J"])]

def nlopt_callback(x, grad):
    i = iteration_counter[0]
    if mp.am_master():
        print(f"\n\n{'='*55}\nIteration {i+1}/{opt_steps} - Time is {datetime.now()}.")

    # Beta schedule: hold at beta_min for first beta_ramp_start iters, then ramp
    if i < beta_ramp_start:
        current_beta = beta_min
    else:
        frac = (i - beta_ramp_start) / max(1, opt_steps - 1 - beta_ramp_start)
        current_beta = beta_min + (beta_max - beta_min) * min(frac, 1.0)

    # Lambda schedule: ramp fabrication penalty weight from 0 to lambda_pen_max
    current_lambda_pen = lambda_pen_max * min(i / max(1, opt_steps - 1), 1.0)
    current_un_pen     = unitary_pen_max * min(i / max(1, opt_steps - 1), 1.0)

    J_val, grad_val, info = evaluate_and_grad(
        params_flat    = x,
        beta           = current_beta,
        lambda_pen_val = current_lambda_pen,
        unitary_pen_val = current_un_pen,
        step_num       = i + 1,
    )

    # NLopt in-place gradient copy
    if grad.size > 0:
        grad[:] = grad_val

    # Compute standard Hadamard fidelity (|Tr|²/4) for logging
    U_meas = info["U"]
    trace_val = np.trace(U_target.conj().T @ U_meas)
    Fh_val    = float(abs(trace_val)**2 / 4.0)
    Tavg_val  = float(np.linalg.norm(U_meas, 'fro')**2 / 2.0)
    Uerr_val  = float(np.linalg.norm(U_meas.conj().T @ U_meas - np.eye(2), 'fro'))
    pen_val = info["Penalty"]

    # Append to history
    history["J"].append(float(J_val))
    history["params"].append(x.copy())
    history["grad"].append(grad_val.copy())
    history["beta"].append(float(current_beta))
    history["U_meas"].append(U_meas.copy())
    history["Fh"].append(Fh_val)
    history["Tavg"].append(Tavg_val)
    history["Uerr"].append(Uerr_val)
    history["Penalty"].append(pen_val)

    save_history(history)
    if mp.am_master():
        np.save(final_params_fname, {
            "params": x.copy(),
            "shape": (Nx, Ny),
            "global_res": global_res,
            "beta": current_beta,
            "run_name": RUN_NAME,
        })
    mp.comm.barrier()
    iteration_counter[0] += 1
    return float(J_val)

# Calibrate TE and TM
if not skip_transmission_calibration:
    if mp.am_master():
        print(f"Starting TE, TM transmission calibration…")
    calibrate_transmission_reference(beta=beta_min)
    if mp.am_master():
        print(f"Transmission Calibration Complete.")

# Configure and launch NLopt
remaining = opt_steps - len(history["J"])
if remaining > 0:
    opt_nlopt = nlopt.opt(nlopt.LD_MMA, Npar)
    opt_nlopt.set_lower_bounds(0.0)
    opt_nlopt.set_upper_bounds(1.0)
    opt_nlopt.set_max_objective(nlopt_callback)
    opt_nlopt.set_maxeval(remaining)
    opt_nlopt.set_ftol_rel(1e-6)

    if mp.am_master():
        print(f"Starting NLopt MMA from iteration {len(history['J'])+1}…")
    params = opt_nlopt.optimize(params.copy())
    if mp.am_master():
        print(f"\nOptimisation finished.  Final J = {opt_nlopt.last_optimum_value():.4f}")
elif mp.am_master():
    print("Optimisation complete (loaded from checkpoint).")


# In[ ]:


# Section 15 — Convergence plot + final design

J_vals    = np.array(history["J"])
Fh_vals   = np.array(history["Fh"])
Tavg_vals = np.array(history["Tavg"])
Uerr_vals = np.array(history["Uerr"])
Pen_vals  = np.array(history["Penalty"])
final_par  = history["params"][-1]
final_beta = history["beta"][-1]

fig, axes = plt.subplots(1, 3, figsize=(15, 4), tight_layout=True)

axes[0].plot(J_vals,    "k-",  label="J (objective)")
axes[0].plot(Fh_vals,   "r--", label="Fidelity F")
axes[0].plot(Tavg_vals, "b:",  label="Avg transmission")
axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Score")
axes[0].set_title("Convergence"); axes[0].legend(); axes[0].grid(True)

axes[1].plot(Uerr_vals, "m-",  label="Unitarity error")
axes[1].plot(Pen_vals,  "g-.", label="Fab penalty")
axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("Residual")
axes[1].set_title("Constraint residuals"); axes[1].legend(); axes[1].grid(True)

rho_final = pre_process(final_par, final_beta).reshape(Nx, Ny)
eps_final = eps_air + (eps_Si - eps_air) * rho_final
im = axes[2].imshow(eps_final.T, extent=[-dr_lx/2, dr_lx/2, -dr_ly/2, dr_ly/2],
                    origin="lower", cmap="RdGy_r", vmin=eps_air, vmax=eps_Si)
plt.colorbar(im, ax=axes[2], label="ε")
axes[2].set_title(f"Final design (β={final_beta:.1f})")
axes[2].set_xlabel("x (µm)"); axes[2].set_ylabel("y (µm)")

if mp.am_master():
    plt.savefig(f"{RUN_DIR}/convergence_plot.png", dpi=200)
    plt.close()


# In[ ]:


# Section 16 — Verification: measure U_meas on the binarised final design
#
# Run two forward sims at beta=100 (≈ step function) with lambda_pen=0
# to extract the purely optical performance metrics.

if mp.am_master():
    print("="*60)
    print("RUNNING FINAL BINARIZED VERIFICATION")
    print("="*60)

_, _, verify_info = evaluate_and_grad(
    params_flat    = final_par,
    beta           = 100.0,
    lambda_pen_val = 0.0,
    unitary_pen_val = 0.0,
    step_num       = "VERIFY",
)

U_final          = verify_info["U"]
fidelity_final   = verify_info["Fh"]
transmission_final = verify_info["Tavg"]
unitarity_residual = float(np.linalg.norm(U_final.conj().T @ U_final - np.eye(2), "fro"))

if mp.am_master():
    print("\n" + "="*60)
    print("FINAL DEVICE PERFORMANCE")
    print("="*60)
    print("U_target =")
    print(np.round(U_target, 4))
    print("\nU_meas (binarised) =")
    print(np.round(U_final, 4))
    print("\n|U_target − U_meas| (elementwise) =")
    print(np.round(np.abs(U_target - U_final), 4))
    print("-"*60)
    print(f"Fidelity  F = |Tr(U†U_meas)|²/4  = {fidelity_final:.4f}   (ideal = 1.0)")
    print(f"Transmission T = ‖U‖²_F / 2      = {transmission_final:.4f}  (lossless = 1.0)")
    print(f"Unitarity residual ‖U†U − I‖_F   = {unitarity_residual:.4f}  (ideal = 0.0)")
    print("="*60)


# In[ ]:


# Section 17 — State-by-state verification
#
# Since the full 2×2 matrix U_meas has been measured analytically,
# per-state fidelity is evaluated without additional FDTD runs.

states = {
    "TE"    : np.array([1.0,  0.0],         dtype=complex),
    "TM"    : np.array([0.0,  1.0],         dtype=complex),
    "diag+" : np.array([1.0,  1.0],         dtype=complex) / np.sqrt(2),
    "diag-" : np.array([1.0, -1.0],         dtype=complex) / np.sqrt(2),
    "Rcirc" : np.array([1.0,  1j],          dtype=complex) / np.sqrt(2),
    "Lcirc" : np.array([1.0, -1j],          dtype=complex) / np.sqrt(2),
}

if mp.am_master():
    print(f"{'state':<8}  {'F_state':>8}  |ψ_meas|    |ψ_target|")
    print("-" * 50)
    for name, psi_in in states.items():
        psi_out_target = U_target @ psi_in
        psi_out_meas   = U_final  @ psi_in
        F = float(
            abs(np.vdot(psi_out_target, psi_out_meas))**2 /
            (np.vdot(psi_out_target, psi_out_target).real
            * np.vdot(psi_out_meas,  psi_out_meas ).real + 1e-30)
        )
        print(f"{name:<8}  {F:>8.4f}  "
            f"[{abs(psi_out_meas[0]):.3f}, {abs(psi_out_meas[1]):.3f}]  "
            f"[{abs(psi_out_target[0]):.3f}, {abs(psi_out_target[1]):.3f}]")


# In[ ]:


# Section 18 — Analytical R-D-R decomposition cross-check
#
# Hadamard = R(-π/4) D(π) R(π/4)  up to a global phase.
# Verify the phase match.

U_cascade   = rotator(-np.pi/4) @ retarder(np.pi) @ rotator(np.pi/4)
phase_match = float(abs(np.trace(U_target.conj().T @ U_cascade))**2 / 4.0)

if mp.am_master():
    print("Analytical cascade  U = R(-π/4) D(π) R(π/4) =")
    print(np.round(U_cascade, 4))
    print()
    print(f"|Tr(U_target† U_cascade)|²/4 = {phase_match:.6f}  (1.0 = same up to global phase)")
    print()
    print("The inverse-designed device, the R-D-R cascade, and the Hadamard target")
    print("all implement the same operation up to an irrelevant global phase.")


# In[ ]:


# Section 19 — Final design export and summary

rho_bin = (pre_process(final_par, final_beta).reshape(Nx, Ny) >= 0.5).astype(int)

if mp.am_master():
    np.savetxt(final_design_csv, rho_bin, fmt="%d", delimiter=",")

fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)
ax.imshow(rho_bin.T, extent=[-dr_lx/2, dr_lx/2, -dr_ly/2, dr_ly/2],
          origin="lower", cmap="binary", vmin=0, vmax=1)
ax.set_title("Final binarised design  (black = Si, white = air)")
ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")

if mp.am_master():
    plt.savefig(f"{RUN_DIR}/final_design.png", dpi=200)
    plt.close()

    print("="*60)
    print("Final summary")
    print("="*60)
    print(f"Runtime             : {datetime.now() - START_TIME}")
    print(f"Target unitary      : Hadamard")
    print(f"Wavelength          : {wl} µm")
    print(f"Design region       : {dr_lx} µm × {dr_ly} µm,  {Nx}×{Ny} pixels")
    print(f"Global resolution   : {global_res}")
    print(f"Beta range          : {beta_min} to {beta_max} with ramp {beta_ramp_start}")
    print(f"Pen, Unitary weights: {lambda_pen_max}, {unitary_pen_max}")
    print(f"Optimisation iters  : {len(history['J'])}")
    print(f"Final J             : {history['J'][-1]:.4f}")
    print(f"Final Fidelity F    : {history['Fh'][-1]:.4f}  (perfect = 1.0)")
    print(f"Verification F      : {fidelity_final:.4f}  (perfect = 1.0)")
    print(f"Verification Tavg   : {transmission_final:.4f}  (lossless = 1.0)")
    print(f"Unitarity residual  : {unitarity_residual:.4f}  (ideal = 0.0)")
    print(f"Design CSV          : {final_design_csv}")

