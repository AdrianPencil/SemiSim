"""
semisim/utils/diagnostics.py

Targeted, low-noise diagnostics to understand why a solve fails.
Import and call these from solvers/workflows when debug=True.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _fmt_range(x: np.ndarray, name: str) -> str:
    if x.size == 0:
        return f"{name}: (empty)"
    return f"{name}∈[{np.min(x):+.3e},{np.max(x):+.3e}]"


def log_state_summary(
    *,
    phi: np.ndarray,
    n: Optional[np.ndarray] = None,
    p: Optional[np.ndarray] = None,
    resid: Optional[np.ndarray] = None,
    prefix: str = "[diag]",
) -> None:
    """Print compact ranges for core fields."""
    msg = [prefix, _fmt_range(phi, "φ")]
    if n is not None:
        msg.append(_fmt_range(n, "n"))
    if p is not None:
        msg.append(_fmt_range(p, "p"))
    if resid is not None:
        msg.append(f"||res||_inf={float(np.linalg.norm(resid, ord=np.inf)):.3e}")
    print(" | ".join(msg))


def check_neutrality(
    *,
    rho_vol_Cm3: np.ndarray,
    sigma_node_Cm2: np.ndarray,
    Vi: np.ndarray,
    tol_abs_C: float = 1e-18,
    prefix: str = "[diag]",
) -> None:
    """
    Report total charge in the domain (useful at equilibrium).
    """
    total_vol = float(np.sum(rho_vol_Cm3 * Vi))
    total_sheet = float(np.sum(sigma_node_Cm2))
    total = total_vol + total_sheet
    print(f"{prefix} charge audit: Q_vol={total_vol:+.3e} C, Q_sheet={total_sheet:+.3e} C, "
          f"Q_total={total:+.3e} C | ok={abs(total) < tol_abs_C}")


def summarize_masks(
    *,
    carrier_mask: Optional[np.ndarray],
    prefix: str = "[diag]",
) -> None:
    if carrier_mask is None:
        print(f"{prefix} carrier mask: None (all nodes active)")
        return
    active = int(np.count_nonzero(carrier_mask))
    total = int(carrier_mask.size)
    print(f"{prefix} carrier mask: {active}/{total} nodes active ({100.0*active/total:.1f}%)")

def log_homotopy_step(
    *,
    s: float,
    accepted: bool,
    converged: bool,
    res_norm: float,
    prev_res_norm: float,
    enable_carriers: bool,
    stats_mode: str,
    enable_2deg: bool,
    lam_min: float,
    dz_min: float,
    max_eta: float,
    prefix: str = "[hom]",
    solver: str | None = None,
) -> None:
    """
    Compact log for each homotopy step. Called by solver/homotopy.py.
    """
    prev_txt = f"{prev_res_norm:.3e}" if np.isfinite(prev_res_norm) else "nan"
    solver_txt = f" sol={solver}" if solver else ""
    print(
        f"{prefix} s={s:g}{solver_txt} | accept={accepted} conv={converged} | "
        f"||r||_inf={res_norm:.3e} (prev={prev_txt}) | "
        f"carriers={enable_carriers} stats={stats_mode} 2DEG={enable_2deg} | "
        f"λD_min={lam_min:.3e} dz_min={dz_min:.3e} | maxη={max_eta:.2f}"
    )

def log_solver_start(
    *,
    solver: str,
    res_inf: float,
    phi_min: float,
    phi_max: float,
    damping: float,
    prefix: str = "[sol]",
) -> None:
    print(
        f"{prefix} {solver} start | ||res||_inf={res_inf:.3e} | "
        f"φ∈[{phi_min:+.3e},{phi_max:+.3e}] V | damping={damping:.2e}"
    )


def log_solver_iter(
    *,
    solver: str,
    it: int,
    res_inf: float,
    damping: float,
    max_dphi: float,
    phi_min: float,
    phi_max: float,
    prefix: str = "[sol]",
) -> None:
    print(
        f"{prefix} {solver} iter {it:02d} | ||res||_inf={res_inf:.3e} | "
        f"damping={damping:.2e} | max|Δφ|={max_dphi:.3e} V | "
        f"φ∈[{phi_min:+.3e},{phi_max:+.3e}] V"
    )


def log_solver_backtrack_fail(
    *,
    solver: str,
    it: int,
    res_inf: float,
    min_damping: float,
    prefix: str = "[sol]",
) -> None:
    print(
        f"{prefix} {solver} iter {it:02d} | line-search failed | "
        f"||res||_inf={res_inf:.3e} | damping_min={min_damping:.1e}"
    )


def log_convergence_summary(
    *,
    solver: str,
    converged: bool,
    iters: int,
    res_inf: float,
    prefix: str = "[sol]",
) -> None:
    print(
        f"{prefix} {solver} done | converged={converged} | iters={iters} | "
        f"||res||_inf={res_inf:.3e}"
    )

def log_contacts_summary(contact_left, contact_right, *, prefix: str = "[diag]") -> None:
    def _one(c):
        if c is None:
            return "None"
        if getattr(c, "kind", "") == "schottky":
            m = c.model
            return (f"Schottky(mu_M set={m.mu_M_J is not None}, "
                    f"phi_Bn={getattr(m,'phi_Bn_J',None)}, phi_Bp={getattr(m,'phi_Bp_J',None)})")
        if getattr(c, "kind", "") == "ohmic":
            m = c.model
            return f"Ohmic(Sn={getattr(m,'S_n_m_per_s',None)}, Sp={getattr(m,'S_p_m_per_s',None)})"
        return str(getattr(c, "kind", "unknown"))
    print(f"{prefix} contacts | left={_one(contact_left)} | right={_one(contact_right)}")

def log_traps_summary(
    *,
    bulk_traps=None,
    interface_traps=None,
    prefix: str = "[diag]"
) -> None:
    """Tiny summary: counts and whether any entries exist."""
    nb = (len(getattr(bulk_traps, "traps", [])) if bulk_traps is not None else 0)
    ni_d = (len(getattr(interface_traps, "discrete", [])) if interface_traps is not None else 0)
    ni_s = (len(getattr(interface_traps, "spectra", [])) if interface_traps is not None else 0)
    print(f"{prefix} traps | bulk={nb} | iface(discrete)={ni_d} | iface(spectra)={ni_s}")
