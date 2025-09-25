# Fully self-contained, extensible plotting script for PA–IA vs Visibility costs
# Uses paper-style operation counts and Δt = (Nb * Nf) / (2B) for real-sampled inputs.
#
# How to extend:
#  - Add another survey dict to SURVEYS below (set Na, B_hz, Nf, and two time resolutions).
#  - Optionally change NB_MAX or y-axis limits.
#
# Requested styling:
#  - PA–IA curve in RED; Visibility curve in GREEN
#  - No grid lines
#  - Left column = lower time resolution (longer Δt); Right column = higher time resolution (shorter Δt)
#  - X-axis: 0 → 1600 beams
#  - Y-axis: log scale with ticks at 0.1, 1, 10, 100, 1000, 10000 (arbitrary G-op units)

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Core math (paper-style terms)
# ------------------------------
def compute_Nb_from_dt(dt_sec: float, B_hz: float, Nf: int):
    """
    Compute Nb from Δt = Nb * Nf / (2B)  =>  Nb = Δt * 2B / Nf
    Returns (Nb_rounded_int, Nb_float_exact).
    """
    Nb = dt_sec * 2.0 * B_hz / Nf
    Nb_int = int(max(1, round(Nb)))
    return Nb_int, Nb

def ops_pa_ia(Na: int, NB: int, Nb: int, Nf: int):
    """
    PA–IA (voltage-route) operations per second for one integration window.
    Mirrors the paper's component list up to constant factors.
    """
    # 1) FFT
    fft = 5.0 * Na * Nb * Nf * np.log2(Nf)
    # 2) Fringe + frac delay + beam steering (per beam)
    steer = NB * Na * Nb * Nf
    # 3) PA beam formation (sum antennas, square, integrate)
    pa = NB * Nf * (Na * Nb + (Nb - 1))
    # 4) IA beam formation (square per ant, sum, integrate)
    ia = Nb * Na * Nf + (Na - 1) * (Nb - 1) * Nf
    # 5) Final PA–IA combine
    final_sub = NB * Nf
    return fft + steer + pa + ia + final_sub

def ops_visibility(Na: int, NB: int, Nb: int, Nf: int):
    """
    Visibility (post-correlation) operations per second for one integration window.
    - Correlation ~ O(Na^2) per (block, channel)
    - Per-beam steering + summation each ~ O(Na^2) per (channel)
    """
    # 1) FFT
    fft = 5.0 * Na * Nb * Nf * np.log2(Nf)
    # 2) Pointing-center fringe/fractional-delay
    center_corr = Na * Nb * Nf
    # 3) Correlation (counting mults+adds together as ~ Na*(Na-1))
    corr = Nb * Nf * (Na * (Na - 1))
    # 4) Per-beam steering (vis domain)
    beam_steer = NB * Nf * (Na * (Na - 1) / 2.0)
    # 5) Per-beam visibility sum to PC power
    beam_sum = NB * Nf * (Na * (Na - 1) / 2.0)
    return fft + center_corr + corr + beam_steer + beam_sum

# ------------------------------
# Plotting helper
# ------------------------------
def plot_two_resolutions(ax_left, ax_right, *, survey_name: str, Na: int, B_hz: float, Nf: int,
                         dt_low: float, dt_high: float, NB_max: int = 1600,
                         y_min: float = 1e-1, y_max: float = 1e4, scale: float = 1e9):
    """
    Make two panels for one survey:
      - Left  : lower time resolution (longer Δt)  -> dt_low
      - Right : higher time resolution (shorter Δt)-> dt_high
    """
    def render(ax, dt_sec: float, title_suffix: str):
        Nb_int, Nb_flt = compute_Nb_from_dt(dt_sec, B_hz, Nf)
        NB_vals = np.arange(1, NB_max + 1, dtype=int)
        pa_vals = np.array([ops_pa_ia(Na, nb_, Nb_int, Nf) for nb_ in NB_vals])
        vis_vals = np.array([ops_visibility(Na, nb_, Nb_int, Nf) for nb_ in NB_vals])

        ax.plot(NB_vals, pa_vals/scale, label="PA–IA (voltage)", color="red", linewidth=1.7)
        ax.plot(NB_vals, vis_vals/scale, label="Visibility (post-corr)", color="green", linewidth=1.7)
        ax.set_title(f"{survey_name}: Δt = {title_suffix}", fontsize=11)
        ax.set_xlabel("Number of formed beams (N_B)")
        ax.set_ylabel("Relative operations (×1e9 units)")
        ax.set_yscale('log')
        ax.set_xlim(0, NB_max)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks([1e-1, 1, 1e1, 1e2, 1e3, 1e4])
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        # No grid lines as requested
        # Legend
        ax.legend(fontsize=9, loc="lower right")
        # Annotate Nb used
        txt = f"N_b ≈ {Nb_flt:.2f} ⇒ {Nb_int} blocks\nN_f={Nf}, B={B_hz/1e6:.0f} MHz, N_a={Na}"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, lw=0.5))

    # Left: lower resolution (longer Δt)
    render(ax_left,  dt_low,  f"{dt_low*1e3:.3f} ms" if dt_low >= 1e-3 else f"{dt_low*1e6:.2f} μs")
    # Right: higher resolution (shorter Δt)
    render(ax_right, dt_high, f"{dt_high*1e3:.3f} ms" if dt_high >= 1e-3 else f"{dt_high*1e6:.2f} μs")

# ------------------------------
# Scenarios (easy to extend)
# ------------------------------
SURVEYS = [
    # Top row (GMRT-like)
    dict(name="GMRT-like", Na=30,  B_hz=200e6, Nf=4096,
         dt_low=1.31e-3, dt_high=163.84e-6),
    # Bottom row (SKA1-Mid-like)
    dict(name="SKA1-Mid-like", Na=256, B_hz=300e6, Nf=4096,
         dt_low=2.048e-3, dt_high=64e-6),
]

# SURVEYS = [
#     # Top row (CASM-256)
#     dict(name="CASM-256", Na=256,  B_hz=93e6, Nf=3072,
#          dt_low=1024e-6, dt_high=2048e-6),
#     # Bottom row (DSA-2000)
#     dict(name="DSA-2000", Na=2000, B_hz=250e6, Nf=8192,
#          dt_low=1024e-6, dt_high=2048e-6),
# ]


# ------------------------------
# Build figure: rows = surveys, columns = [low Δt, high Δt]
# ------------------------------
fig, axes = plt.subplots(len(SURVEYS), 2, figsize=(12, 8), constrained_layout=True)

for row_idx, survey in enumerate(SURVEYS):
    ax_left  = axes[row_idx, 0] if len(SURVEYS) > 1 else axes[0]
    ax_right = axes[row_idx, 1] if len(SURVEYS) > 1 else axes[1]
    plot_two_resolutions(
        ax_left, ax_right,
        survey_name=survey["name"],
        Na=survey["Na"], B_hz=survey["B_hz"], Nf=survey["Nf"],
        dt_low=survey["dt_low"], dt_high=survey["dt_high"],
        NB_max=2000, y_min=1e-1, y_max=1e4, scale=1e9
    )

# Save and show
out_path = "pc_beam_costs_gmrt_ska.png"
plt.savefig(out_path, dpi=150)
plt.show()

print(f"Saved figure to {out_path}")
