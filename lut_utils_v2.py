"""
lut_utils_v2.py
================
LUT query engine for the GF180MCU 3.3V master LUT (lut_nmos.csv / lut_pmos.csv).

Replaces the legacy single-key (L, gm/Id) → I/W lookup with a full multi-key
interpolator over (L, V*, Vds) that supports the complete gm/Id design flow:
  - Step 4: pick L given target gm/gds and K = Vds/V*
  - Step 6: pick W given L, Id, V*
  - General query: full op-point readout at any (L, V*, Vds)

Design rules (v0.2 SIZING_FLOW.md):
  - All public functions return DATA (dicts/lists), not formatted text.
  - The MCP tool layer wraps these and Claude decides presentation.
  - Includes physical sanity flags (saturation, range warnings) so the agent
    can recognize unreliable points without re-deriving the physics.

Loading:
  >>> from lut_utils_v2 import LUT
  >>> lut = LUT.load_default()    # reads gmid_LUT/lut_nmos.csv + lut_pmos.csv
  >>> r = lut.query_op_point('nmos', L=0.7e-6, V_star=0.159, Vds=1.65)
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


# ---------------------------------------------------------------------------
# Constants / conventions
# ---------------------------------------------------------------------------

DEFAULT_LUT_DIR = Path(__file__).parent / "gmid_LUT"

# Physical safety rails for GF180MCU 3.3V devices (from PDK).
PROCESS_LIMITS = {
    "L_min": 0.28e-6,
    "L_max": 5.0e-6,           # we don't have data beyond ~3um; reject beyond 5
    "W_min": 0.22e-6,
    "V_star_min": 0.04,        # below this we're in deep weak inversion; gm/Id flat
    "V_star_max": 1.5,         # above this is unphysical for sizing decisions
    "K_min_for_saturation": 1.0,
}

DeviceType = Literal["nmos", "pmos"]


# ---------------------------------------------------------------------------
# Data containers (everything returned to MCP tools is a dict; these are the
# typed handles internally)
# ---------------------------------------------------------------------------

@dataclass
class OpPoint:
    """Full operating-point readout at one (L, V*, Vds)."""
    L: float
    V_star: float
    Vds: float
    Vgs_estimate: float          # back-solved Vgs at this V* (for debug / verification)
    vth: float
    vdsat: float
    id_over_w: float             # A/m
    gm_id: float                 # V^-1
    gm_over_gds: float           # intrinsic gain at this Vds
    ft: float                    # Hz
    K: float                     # = Vds / V_star
    in_saturation: bool          # Vds >= Vdsat
    in_lut_range: bool           # within sweep grid (no extrapolation)
    notes: list[str]             # any sanity-check observations

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class LCandidate:
    """One viable L choice for a given target intrinsic gain + K."""
    L: float
    gm_over_gds_at_K: float      # achievable at the requested K
    ft_at_K: float               # Hz, at this L and K
    area_proxy: float            # = L (channel length itself is the area driver, dummy)
    pvt_margin_flag: Optional[str]  # 'tight', 'comfortable', None
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# LUT engine
# ---------------------------------------------------------------------------

class LUT:
    """Loads the master LUT CSVs and provides physics-aware queries."""

    def __init__(self, nmos_df: pd.DataFrame, pmos_df: pd.DataFrame):
        self._raw = {"nmos": nmos_df, "pmos": pmos_df}

        # Pre-extract sweep grids for each device.
        self._grids = {}            # type: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
        self._interps = {}          # type: dict[tuple[str, str], RegularGridInterpolator]

        for dev, df in self._raw.items():
            df = df.dropna(subset=["V_star", "gm_id", "id_over_w", "gm_over_gds", "ft"])
            Ls   = np.sort(df["L"].unique())
            Vgss = np.sort(df["Vgs"].unique())
            Vdss = np.sort(df["Vds"].unique())
            self._grids[dev] = (Ls, Vgss, Vdss)

            # Build a regular grid: indexed by (L, Vgs, Vds), value = column.
            # Some grid cells are NaN (subthreshold); we'll fill with nearest-valid
            # for interpolation, but flag any query that falls into a NaN region.
            shape = (len(Ls), len(Vgss), len(Vdss))
            for col in ["id", "gm", "gds", "cgg", "vth", "vdsat",
                        "V_star", "gm_id", "id_over_w", "gm_over_gds", "ft", "K"]:
                grid = np.full(shape, np.nan)
                lut_local = df[df[col].notna()]
                for _, row in lut_local.iterrows():
                    i = np.searchsorted(Ls,   row["L"])
                    j = np.searchsorted(Vgss, row["Vgs"])
                    k = np.searchsorted(Vdss, row["Vds"])
                    if i < len(Ls) and j < len(Vgss) and k < len(Vdss):
                        grid[i, j, k] = row[col]
                # For interpolation we need NaN-free; replace NaN with column min
                # (subthreshold regions). We'll separately track validity.
                if np.isnan(grid).any():
                    grid_filled = np.where(np.isnan(grid),
                                           np.nanmin(grid) if col in ("id", "gm", "id_over_w")
                                           else np.nanmean(grid),
                                           grid)
                else:
                    grid_filled = grid
                self._interps[(dev, col)] = RegularGridInterpolator(
                    (Ls, Vgss, Vdss),
                    grid_filled,
                    method="linear",
                    bounds_error=False,
                    fill_value=None,
                )

    # ----- loaders -----

    @classmethod
    def load_default(cls, lut_dir: Optional[Path] = None) -> "LUT":
        d = Path(lut_dir) if lut_dir else DEFAULT_LUT_DIR
        nmos = pd.read_csv(d / "lut_nmos.csv")
        pmos = pd.read_csv(d / "lut_pmos.csv")
        return cls(nmos, pmos)

    # ----- internal helpers -----

    def _vgs_for_vstar(self, dev: DeviceType, L: float, V_star: float, Vds: float) -> float:
        """
        The LUT is indexed by Vgs (the sweep axis), but designers think in V*.
        For a fixed (L, Vds), V* is monotone-decreasing in Vgs (above threshold).
        Bisect on the V_star interpolator to find the Vgs that gives the target V*.
        """
        Ls, Vgss, Vdss = self._grids[dev]

        # Sweep V_star along Vgs at this L, Vds
        Vgs_grid = np.linspace(Vgss[0], Vgss[-1], 200)
        pts = np.column_stack([np.full_like(Vgs_grid, L),
                               Vgs_grid,
                               np.full_like(Vgs_grid, Vds)])
        Vstar_curve = self._interps[(dev, "V_star")](pts)
        # The curve typically goes from ~1.5 down to ~0.05 as Vgs increases.
        # Use np.interp on the monotone-decreasing curve (need to flip).
        order = np.argsort(Vstar_curve)
        return float(np.interp(V_star, Vstar_curve[order], Vgs_grid[order]))

    def _interp_at_vgs(self, dev: DeviceType, L: float, Vgs: float, Vds: float) -> dict:
        out = {}
        pt = np.array([[L, Vgs, Vds]])
        for col in ["id", "gm", "gds", "cgg", "vth", "vdsat",
                    "V_star", "gm_id", "id_over_w", "gm_over_gds", "ft", "K"]:
            out[col] = float(self._interps[(dev, col)](pt)[0])
        return out

    def _range_check(self, dev: DeviceType,
                     L: float, V_star: float, Vds: float) -> tuple[bool, list[str]]:
        notes = []
        in_range = True
        Ls, Vgss, Vdss = self._grids[dev]
        if not (Ls[0] <= L <= Ls[-1]):
            in_range = False
            notes.append(f"L={L*1e6:.3f}um outside sweep range "
                         f"[{Ls[0]*1e6:.3f}, {Ls[-1]*1e6:.3f}]um")
        if not (PROCESS_LIMITS["V_star_min"] <= V_star <= PROCESS_LIMITS["V_star_max"]):
            in_range = False
            notes.append(f"V*={V_star:.3f}V outside reasonable range "
                         f"[{PROCESS_LIMITS['V_star_min']}, {PROCESS_LIMITS['V_star_max']}]V")
        if not (Vdss[0] <= Vds <= Vdss[-1]):
            in_range = False
            notes.append(f"Vds={Vds:.2f}V outside sweep range "
                         f"[{Vdss[0]:.2f}, {Vdss[-1]:.2f}]V")
        return in_range, notes

    # ----- public API: 3 query functions (correspond to MCP tools 4, 5, 6) -----

    def query_op_point(self, type: DeviceType, L: float, V_star: float,
                       Vds: Optional[float] = None) -> dict:
        """
        General op-point readout. Used by MCP tool `query_op_point`.

        Default Vds: 2 * V_star (the common K=2 saturation-margin choice).
        Returns a dict; the MCP tool wraps this without reformatting.
        """
        if Vds is None:
            Vds = 2.0 * V_star

        in_range, range_notes = self._range_check(type, L, V_star, Vds)

        try:
            Vgs = self._vgs_for_vstar(type, L, V_star, Vds)
            data = self._interp_at_vgs(type, L, Vgs, Vds)
        except Exception as e:
            return {
                "error": f"interpolation failed: {e}",
                "L": L, "V_star": V_star, "Vds": Vds,
            }

        notes = list(range_notes)
        in_sat = Vds >= data["vdsat"]
        if not in_sat:
            notes.append(f"NOT in saturation: Vds={Vds:.3f} < Vdsat={data['vdsat']:.3f}; "
                         f"gm/gds value is unreliable")

        op = OpPoint(
            L=L, V_star=V_star, Vds=Vds,
            Vgs_estimate=Vgs,
            vth=data["vth"], vdsat=data["vdsat"],
            id_over_w=data["id_over_w"],
            gm_id=data["gm_id"],
            gm_over_gds=data["gm_over_gds"],
            ft=data["ft"],
            K=Vds / V_star if V_star > 0 else float("inf"),
            in_saturation=in_sat,
            in_lut_range=in_range,
            notes=notes,
        )
        return op.to_dict()

    def lookup_W_from_current(self, type: DeviceType, L: float, Id: float,
                              V_star: float, Vds: Optional[float] = None) -> dict:
        """
        Garrett flow step 6: given L, Id (per-finger drain current), V* (chosen by
        designer or pinned by gm spec), return W and the implied op-point.

        Used by MCP tool `lookup_W_from_current`.
        """
        if Vds is None:
            Vds = 2.0 * V_star

        op = self.query_op_point(type, L, V_star, Vds)
        if "error" in op:
            return op

        I_star = op["id_over_w"]                # A/m
        if I_star <= 0:
            return {"error": f"I* <= 0 at this op-point", "op": op}

        W = Id / I_star                          # m
        notes = list(op["notes"])
        if W < PROCESS_LIMITS["W_min"]:
            notes.append(f"W={W*1e6:.3f}um BELOW PDK minimum {PROCESS_LIMITS['W_min']*1e6}um; "
                         f"consider lower V* (smaller I*) or smaller Id")

        return {
            "W": W,
            "W_um": W * 1e6,
            "L": L, "L_um": L * 1e6,
            "Id": Id, "Id_uA": Id * 1e6,
            "V_star": V_star,
            "Vds": Vds,
            "I_star": I_star, "I_star_uA_per_um": I_star * 1e6,
            "ft": op["ft"], "ft_GHz": op["ft"] / 1e9,
            "gm_id": op["gm_id"],
            "gm": Id * op["gm_id"],
            "gm_uS": Id * op["gm_id"] * 1e6,
            "gm_over_gds": op["gm_over_gds"],
            "vth": op["vth"],
            "vdsat": op["vdsat"],
            "in_saturation": op["in_saturation"],
            "in_lut_range": op["in_lut_range"],
            "notes": notes,
        }

    def lookup_L_from_intrinsic_gain(self, type: DeviceType,
                                     target_gain: float,
                                     K: float = 2.0,
                                     V_star_for_eval: Optional[float] = None) -> dict:
        """
        Garrett flow step 4: given a target gm/gds (intrinsic gain) and a K =
        Vds/V*, return all L's in the sweep, each with the gain achievable
        and a pvt_margin_flag.

        Used by MCP tool `lookup_L_from_intrinsic_gain`.

        Note: gm/gds depends primarily on L (PDF p.20). We sweep all L's,
        evaluate at K = Vds/V*, and rank.

        Args:
            type: 'nmos' or 'pmos'
            target_gain: required gm/gds per device
            K: Vds/V* — default 2.0 (the standard saturation-margin choice)
            V_star_for_eval: at which V* to evaluate the L curve. If None,
                              uses V* = 0.2V (PDF FoM-peak), a reasonable proxy
                              for "typical operating point during sizing".

        Returns dict with all L candidates and the recommended subset.
        """
        if V_star_for_eval is None:
            V_star_for_eval = 0.20    # PDF FoM-peak, see GF180MCU validation
        Vds = K * V_star_for_eval

        Ls, _, _ = self._grids[type]
        candidates = []
        for L in Ls:
            op = self.query_op_point(type, L, V_star_for_eval, Vds)
            if "error" in op:
                continue
            gain_achievable = op["gm_over_gds"]
            ft_at_op = op["ft"]

            # PVT margin flag — heuristic
            if gain_achievable < target_gain * 1.2:
                margin = "tight"     # less than 20% margin
            elif gain_achievable < target_gain * 2.0:
                margin = "comfortable"
            else:
                margin = "ample"

            notes = []
            if not op["in_saturation"]:
                notes.append("not in saturation at this eval point")

            candidates.append(LCandidate(
                L=L,
                gm_over_gds_at_K=gain_achievable,
                ft_at_K=ft_at_op,
                area_proxy=L,
                pvt_margin_flag=margin if gain_achievable >= target_gain else "insufficient",
                notes=notes,
            ).to_dict())

        # Mark candidates that hit the target
        meets_target = [c for c in candidates if c["gm_over_gds_at_K"] >= target_gain]

        result = {
            "target_gm_over_gds": target_gain,
            "K_eval": K,
            "V_star_eval": V_star_for_eval,
            "Vds_eval": Vds,
            "all_candidates": candidates,
            "candidates_meeting_target": meets_target,
            "notes": [],
        }
        if not meets_target:
            best = max(candidates, key=lambda c: c["gm_over_gds_at_K"])
            result["notes"].append(
                f"target gm/gds={target_gain} not achievable on this process at K={K}; "
                f"best is L={best['L']*1e6:.2f}um giving gm/gds={best['gm_over_gds_at_K']:.0f}"
            )
        return result


# ---------------------------------------------------------------------------
# Backward-compatibility shims (so AutoSizer's old imports keep working)
# ---------------------------------------------------------------------------

_GLOBAL_LUT: Optional[LUT] = None


def _get_global_lut() -> LUT:
    global _GLOBAL_LUT
    if _GLOBAL_LUT is None:
        _GLOBAL_LUT = LUT.load_default()
    return _GLOBAL_LUT


def calculate_w_linear_NMOS_pro(aim_L: float, aim_I: float, gmid: float) -> float:
    """Legacy signature from the old lut_utils.py.
    Maps to the new LUT by converting gmid → V* = 2/gmid.
    """
    if gmid <= 0:
        warnings.warn(f"gmid={gmid} invalid; returning W_min")
        return PROCESS_LIMITS["W_min"]
    V_star = 2.0 / gmid
    lut = _get_global_lut()
    r = lut.lookup_W_from_current("nmos", L=aim_L, Id=aim_I, V_star=V_star)
    if "error" in r:
        warnings.warn(f"new LUT failed for L={aim_L}, I={aim_I}, gmid={gmid}: {r['error']}")
        return PROCESS_LIMITS["W_min"]
    return max(r["W"], PROCESS_LIMITS["W_min"])


def calculate_w_linear_PMOS_pro(aim_L: float, aim_I: float, gmid: float) -> float:
    """Legacy signature, PMOS counterpart."""
    if gmid <= 0:
        warnings.warn(f"gmid={gmid} invalid; returning W_min")
        return PROCESS_LIMITS["W_min"]
    V_star = 2.0 / gmid
    lut = _get_global_lut()
    r = lut.lookup_W_from_current("pmos", L=aim_L, Id=aim_I, V_star=V_star)
    if "error" in r:
        warnings.warn(f"new LUT failed for L={aim_L}, I={aim_I}, gmid={gmid}: {r['error']}")
        return PROCESS_LIMITS["W_min"]
    return max(r["W"], PROCESS_LIMITS["W_min"])


# ---------------------------------------------------------------------------
# Smoke test (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    lut = LUT.load_default()
    print("=" * 70)
    print("  lut_utils_v2 smoke test")
    print("=" * 70)

    # Test 1: query_op_point at SIZING_FLOW.md Turn 6 conditions
    print("\n[Test 1] query_op_point: NMOS, L=0.7u, V*=0.159, Vds=1.65")
    r = lut.query_op_point("nmos", L=0.7e-6, V_star=0.159, Vds=1.65)
    for k, v in r.items():
        if isinstance(v, float):
            print(f"  {k:20s} = {v:.4g}")
        else:
            print(f"  {k:20s} = {v}")

    # Test 2: lookup_W_from_current — Turn 6 result
    print("\n[Test 2] lookup_W_from_current: NMOS, L=0.7u, Id=10uA, V*=0.159")
    r = lut.lookup_W_from_current("nmos", L=0.7e-6, Id=10e-6, V_star=0.159)
    for k, v in r.items():
        if isinstance(v, float):
            print(f"  {k:20s} = {v:.4g}")
        else:
            print(f"  {k:20s} = {v}")
    print(f"\n  EXPECTED (from SIZING_FLOW.md Turn 6): W ≈ 6.76 um")

    # Test 3: lookup_L_from_intrinsic_gain — Turn 4 conditions
    print("\n[Test 3] lookup_L_from_intrinsic_gain: NMOS, target gm/gds=200, K=2")
    r = lut.lookup_L_from_intrinsic_gain("nmos", target_gain=200, K=2.0)
    print(f"  target = {r['target_gm_over_gds']}, K = {r['K_eval']}, V* = {r['V_star_eval']}")
    print(f"  {'L[um]':>7} {'gm/gds':>9} {'ft[GHz]':>9} {'margin':>14}")
    for c in r["all_candidates"]:
        print(f"  {c['L']*1e6:7.2f} {c['gm_over_gds_at_K']:9.1f} "
              f"{c['ft_at_K']/1e9:9.2f} {c['pvt_margin_flag']:>14}")
    print(f"\n  EXPECTED (from SIZING_FLOW.md Turn 4):")
    print(f"  L=0.5u → gm/gds≈91 (tight); L=0.7u → ≈148 (comfortable); L=1.0u → ≈210 (ample)")

    # Test 4: backward-compat shim
    print("\n[Test 4] Legacy API: calculate_w_linear_NMOS_pro(L=0.7u, I=10uA, gmid=12.6)")
    W = calculate_w_linear_NMOS_pro(0.7e-6, 10e-6, 12.6)
    print(f"  W = {W*1e6:.3f} um   (should match Test 2)")

    print("\n[Test 5] PMOS mirror at SIZING_FLOW.md Turn 7 conditions")
    print("  query: L=1u, Id=10uA, V*=0.25")
    r = lut.lookup_W_from_current("pmos", L=1.0e-6, Id=10e-6, V_star=0.25)
    print(f"  W = {r['W_um']:.2f} um, gm/gds = {r['gm_over_gds']:.1f}, gm/Id = {r['gm_id']:.2f}")
    print(f"  EXPECTED: W ≈ 11um, gm/gds ≈ 200, gm/Id ≈ 8")

    print("\nDone.")