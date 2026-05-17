import os
import chromadb
from sentence_transformers import SentenceTransformer  # Fallback: bge-m3 本地模型
from typing import List, Dict, Any, Tuple
import sys
from mcp.server.fastmcp import FastMCP
import io
import time
import torch
import traceback
import logging

from multiprocessing import Process
import concurrent.futures
import threading
import requests

mcp = FastMCP("LOCAL")

# --- Database Connection (Assumes DB is already built) ---
db_path = "./database"  # Your Database data path
collection_name = "my_collection"

# --- Embedding Config ---
# SiliconFlow API 优先，失败则 fallback 到本地 bge-m3
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/embeddings"
SILICONFLOW_EMBEDDING_MODEL = os.getenv("SILICONFLOW_EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-8B")
LOCAL_EMBEDDING_MODEL = "BAAI/bge-m3"

# --- Lazy Loading ---
_rag_model = None  # 本地 bge-m3 model (仅 fallback 时加载)
_chroma_client = None
_collection = None
print("ChromaDB client will be loaded on first use. Embeddings: SiliconFlow API (priority) -> local bge-m3 (fallback).")


def get_db_collection():
    """
    Initializes and returns the ChromaDB client and collection on first call.
    """
    global _chroma_client, _collection
    if _collection is None:
        print("Connecting to ChromaDB for the first time...")
        try:
            _chroma_client = chromadb.PersistentClient(path=db_path)
            _collection = _chroma_client.get_collection(name=collection_name)
            print(f"✅ Successfully connected to collection '{collection_name}' with {_collection.count()} documents.")
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Could not connect to ChromaDB collection.")
            print(f"Please run 'build_database.py' script first.")
            print(f"Details: {e}")
            # Raise an exception to stop the tool execution
            raise e
    return _collection


# --- MCP Tools ---

@mcp.tool()
async def rag_query(query: str, num_results: int = 3) -> Dict[str, Any]:
    """
    Queries the analog circuit design knowledge base. The knowledge base contains:
    - gm/Id design tables and methodology for analog OTA sizing
    - Compensation network design rules (Miller, Ahuja, etc.)
    - Layout-aware design considerations and parasitic effects
    - PVT corner analysis methodology

    Call this tool BEFORE making sizing decisions when you need:
    - Recommended gm/Id values for specific transistor roles (input pair, current mirror, etc.)
    - Heuristics for choosing initial L/W ranges
    - Compensation strategy selection guidance
    - Knowledge of process-specific design constraints

    Args:
        query: Natural language question about analog circuit design
        num_results: Number of relevant document chunks to return (default 3)

    Returns:
        Dict with 'results' list, each containing 'content' (document text)
    """
    try:
        collection = get_db_collection()
    except Exception as e:
        return {"results": [{"content": f"Error: Could not connect to DB. {e}"}]}

    print(f"Received query: {query}")

    # --- Embedding: SiliconFlow API 优先，失败 fallback 本地 bge-m3 ---
    query_embedding = None

    # 1) Try SiliconFlow API
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("SILICONFLOW_API_KEY", "")
        if api_key:
            resp = requests.post(
                SILICONFLOW_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json={
                    "model": SILICONFLOW_EMBEDDING_MODEL,
                    "input": [query],
                    "encoding_format": "float",
                },
                timeout=30,
            )
            result = resp.json()
            if "data" in result and len(result["data"]) > 0:
                query_embedding = result["data"][0]["embedding"]
                print("Embedding via SiliconFlow API (Qwen3-VL-Embedding-8B).")
            else:
                print(f"SiliconFlow API returned no data, falling back to local model.")
        else:
            print("SILICONFLOW_API_KEY not set, falling back to local model.")
    except Exception as e:
        print(f"SiliconFlow API failed ({e}), falling back to local model.")

    # 2) Fallback: local bge-m3
    if query_embedding is None:
        global _rag_model
        if _rag_model is None:
            print(f"Loading local embedding model ({LOCAL_EMBEDDING_MODEL})...")
            try:
                _rag_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
                print("Local model loaded.")
            except Exception as e:
                return {"results": [{"content": f"Error: Could not load local embedding model. {e}"}]}
        query_embedding = _rag_model.encode([query], normalize_embeddings=True)[0].tolist()
        print("Embedding via local bge-m3 (fallback).")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=num_results
    )

    # Organize query results
    response = {
        "results": []
    }

    # Check if results are valid and not empty
    if results and results.get('documents') and results['documents'][0]:
        for doc in results['documents'][0]:
            response["results"].append({
                "content": doc
            })
    else:
        print("No documents found for the query.")

    return response


# =============================================================================
#  Vibe Coding Sizing Tools  (v0.2 architecture — see SIZING_FLOW.md)
# =============================================================================
#  Replaces the legacy `find_initial_design` (Stage 1 BO) and `FocalOpt` (Stage 2)
#  with fine-grained, agent-friendly primitives.
#
#  Design rule: tools return DATA (dicts/lists), never pre-formatted text.
#  Claude Code composes presentation from the data shape.
# =============================================================================

import json
from pathlib import Path
from datetime import datetime

# Local module: master LUT engine for GF180MCU 3.3V.
from lut_utils_v2 import LUT, PROCESS_LIMITS

# Lazy-load the LUT (instantiated on first call, cached for server lifetime).
_LUT_INSTANCE = None
def _get_lut() -> LUT:
    global _LUT_INSTANCE
    if _LUT_INSTANCE is None:
        _LUT_INSTANCE = LUT.load_default()
        print(f"✅ Master LUT loaded: NMOS + PMOS, GF180MCU 3.3V")
    return _LUT_INSTANCE


# -----------------------------------------------------------------------------
# Tool 4: lookup_L_from_intrinsic_gain
# -----------------------------------------------------------------------------
@mcp.tool()
async def lookup_L_from_intrinsic_gain(
        type: str,
        target_gm_over_gds: float,
        K: float = 2.0,
        V_star_eval: float = 0.20,
) -> Dict[str, Any]:
    """
    Step 4 of the gm/Id flow: given a required intrinsic gain (gm/gds, also
    written gm·ro) and a saturation factor K = Vds/V*, return all L candidates
    in the LUT sweep with their achievable gain and a PVT margin assessment.

    Use this BEFORE picking W. Call once per device class (input pair, mirror,
    cascode) when you need to decide L based on a gain budget.

    Args:
        type: "nmos" or "pmos"
        target_gm_over_gds: required intrinsic gain per device (e.g. 200 for
            cascoded stages targeting 80+ dB)
        K: Vds/V* — default 2.0 (the standard saturation-margin assumption).
            Set higher (e.g. 4-5) if you have extra Vds headroom in the topology.
        V_star_eval: V* at which to evaluate the L-vs-gain curve. Default 0.20V
            (PDF FoM-peak). Use 0.15 for high-gm input pairs, 0.25 for mirrors.

    Returns:
        Dict with:
            - target_gm_over_gds: echo of input
            - K_eval, V_star_eval, Vds_eval: evaluation point
            - all_candidates: list of {L, gm_over_gds_at_K, ft_at_K,
                area_proxy, pvt_margin_flag, notes} — every L in the sweep
            - candidates_meeting_target: subset that satisfies the gain target
            - notes: warnings if target is unachievable on this process
    """
    if type not in ("nmos", "pmos"):
        return {"error": f"type must be 'nmos' or 'pmos', got '{type}'"}
    try:
        lut = _get_lut()
        return lut.lookup_L_from_intrinsic_gain(
            type=type,
            target_gain=target_gm_over_gds,
            K=K,
            V_star_for_eval=V_star_eval,
        )
    except Exception as e:
        return {"error": f"LUT query failed: {e}", "traceback": traceback.format_exc()}


# -----------------------------------------------------------------------------
# Tool 5: lookup_W_from_current
# -----------------------------------------------------------------------------
@mcp.tool()
async def lookup_W_from_current(
        type: str,
        L: float,
        Id: float,
        V_star: float,
        Vds: float = None,
) -> Dict[str, Any]:
    """
    Step 6 of the gm/Id flow: given L (chosen via lookup_L_from_intrinsic_gain
    or by other means), the per-device drain current Id, and a target V* (which
    is equivalent to gm/Id = 2/V*), compute the device width W from the LUT's
    I* curve, plus the full implied operating point.

    Call this AFTER L is picked. Use the returned W as the initial-design sizing.

    Args:
        type: "nmos" or "pmos"
        L: channel length in meters (e.g. 0.7e-6)
        Id: per-device drain current in amperes (e.g. 10e-6)
        V_star: overdrive proxy = 2·Id/gm, in volts. Typical values:
            - 0.15-0.20V for input pairs / high-gm devices
            - 0.25V for current mirrors (mismatch margin)
            - 0.10V for very low-power / sub-threshold designs
        Vds: device drain-source voltage. If None, defaults to 2·V* (K=2
            saturation margin convention).

    Returns:
        Dict with W (and W_um for human-readable), Id, gm (back-computed),
        ft, gm/Id, gm/gds at this op-point, vth, vdsat, in_saturation flag,
        in_lut_range flag, and notes (e.g. W < W_min warning).
    """
    if type not in ("nmos", "pmos"):
        return {"error": f"type must be 'nmos' or 'pmos', got '{type}'"}
    try:
        lut = _get_lut()
        return lut.lookup_W_from_current(type=type, L=L, Id=Id,
                                         V_star=V_star, Vds=Vds)
    except Exception as e:
        return {"error": f"LUT query failed: {e}", "traceback": traceback.format_exc()}


# -----------------------------------------------------------------------------
# Tool 6: query_op_point
# -----------------------------------------------------------------------------
@mcp.tool()
async def query_op_point(
        type: str,
        L: float,
        V_star: float,
        Vds: float = None,
) -> Dict[str, Any]:
    """
    General-purpose operating-point readout from the LUT. Use this for
    what-if exploration, sanity-checking a proposed design, or computing
    derived quantities without committing to a specific W or Id.

    Returns the full BSIM4 small-signal readout at (L, V*, Vds): vth, vdsat,
    I*=id/W, gm/Id, gm/gds, ft, K=Vds/V*, plus saturation + range flags.

    Use cases:
      - "What ft can I get at L=1u, V*=0.2, Vds=1.65?"
      - "Is L=0.3u, V*=0.1 still in saturation at Vds=0.5?"
      - "What's the intrinsic gain of a PMOS at L=2u, V*=0.25?"

    Args:
        type: "nmos" or "pmos"
        L: channel length in meters
        V_star: 2·Id/gm in volts
        Vds: drain-source voltage. Defaults to 2·V* if None.

    Returns:
        Dict with all op-point quantities and physical sanity flags.
    """
    if type not in ("nmos", "pmos"):
        return {"error": f"type must be 'nmos' or 'pmos', got '{type}'"}
    try:
        lut = _get_lut()
        return lut.query_op_point(type=type, L=L, V_star=V_star, Vds=Vds)
    except Exception as e:
        return {"error": f"LUT query failed: {e}", "traceback": traceback.format_exc()}


# -----------------------------------------------------------------------------
# Tool 1: parse_yaml_spec
# -----------------------------------------------------------------------------
@mcp.tool()
async def parse_yaml_spec(path: str) -> Dict[str, Any]:
    """
    Parse an AutoSizer-format YAML spec file describing a circuit sizing task.
    Returns the parsed dict, with light normalization (unit parsing for
    "10MHz" → 10e6, "2pF" → 2e-12, etc.) so downstream tools can consume it.

    The YAML schema (AutoSizer convention):
        circuit: <name>
        process: <pdk_id>
        vdd: <volts>
        specs:
          GBW: <value with unit>
          CL: <value with unit>
          DC_gain: ">= N dB"
          ...
        constraints:
          L_min: <value>
          L_max: <value>

    Args:
        path: filesystem path to the .yaml spec file

    Returns:
        Dict with parsed spec, normalized to SI units. Includes a 'raw' field
        with the original text for traceability.
    """
    try:
        import yaml
    except ImportError:
        return {"error": "PyYAML not installed; run `pip install pyyaml`"}
    try:
        with open(path, "r") as f:
            raw = f.read()
        parsed = yaml.safe_load(raw)
    except Exception as e:
        return {"error": f"could not read/parse YAML: {e}", "path": path}

    # Light unit normalization on the specs block
    UNIT_MULT = {
        "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3,
        "k": 1e3,   "M": 1e6,   "G": 1e9,
    }
    def _norm(val):
        if isinstance(val, str):
            v = val.strip()
            # Strip common units (Hz, F, V, A, dB, W, °) and return numeric multiplier
            # Strip compound units first (longest match wins), then single units
            for suffix in ("V/us", "V/s", "Hz", "dB", "F", "V", "A", "s", "W"):
                if v.endswith(suffix):
                    v = v[:-len(suffix)]
                    # special compound: V/us is slew rate, base unit is V/s,
                    # so the SI value is the numeric × 1e6
                    if suffix == "V/us":
                        try:
                            return float(v) * 1e6
                        except ValueError:
                            return val
                    break
            # SI prefix
            if v and v[-1] in UNIT_MULT:
                try:
                    return float(v[:-1]) * UNIT_MULT[v[-1]]
                except ValueError:
                    return val
            try:
                return float(v)
            except ValueError:
                return val
        return val

    if isinstance(parsed, dict) and "specs" in parsed:
        for k, v in list(parsed["specs"].items()):
            parsed["specs"][k] = _norm(v)
    if isinstance(parsed, dict) and "constraints" in parsed:
        for k, v in list(parsed["constraints"].items()):
            parsed["constraints"][k] = _norm(v)

    return {
        "spec": parsed,
        "path": str(Path(path).resolve()),
        "raw": raw,
    }


# -----------------------------------------------------------------------------
# Tool 2: compute_design_equations
# -----------------------------------------------------------------------------
@mcp.tool()
async def compute_design_equations(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute the closed-form numbers that fall directly out of a spec:
        gm_in    = 2π · GBW · CL                  (input-pair required gm)
        I_tail   = SR · CL                        (tail current from slew rate)
        I_branch = I_tail / 2                     (per-input-device current)
        gm/Id_in = gm_in / I_branch               (implied current efficiency)
        V*_in    = 2 / (gm/Id_in)                 (implied overdrive)

    These are unique given the spec — no design choice involved.

    Args:
        spec: spec dict (as returned by parse_yaml_spec, or constructed inline)
              Must contain spec['specs'] with GBW, CL, slew_rate (all in SI).

    Returns:
        Dict with derived quantities and the equations used (so the agent
        can explain them to the user).
    """
    import math
    try:
        s = spec.get("specs", spec)  # tolerate either {specs:{...}} or flat dict
        GBW = float(s["GBW"])
        CL  = float(s["CL"])
        SR  = float(s.get("slew_rate", s.get("SR", 0)))
    except KeyError as e:
        return {"error": f"missing spec field: {e}"}
    except (TypeError, ValueError) as e:
        return {"error": f"spec field not numeric: {e}"}

    gm_in = 2 * math.pi * GBW * CL
    I_tail = SR * CL if SR > 0 else None
    I_branch = (I_tail / 2) if I_tail else None
    gm_id_in = (gm_in / I_branch) if I_branch else None
    V_star_in = (2.0 / gm_id_in) if gm_id_in else None

    return {
        "gm_in":      gm_in,
        "gm_in_uS":   gm_in * 1e6,
        "I_tail":     I_tail,
        "I_tail_uA":  (I_tail * 1e6) if I_tail else None,
        "I_branch":   I_branch,
        "I_branch_uA": (I_branch * 1e6) if I_branch else None,
        "gm_id_in":   gm_id_in,
        "V_star_in":  V_star_in,
        "equations_used": {
            "gm_in":    "2 * pi * GBW * CL",
            "I_tail":   "SR * CL",
            "I_branch": "I_tail / 2",
            "gm/Id":    "gm_in / I_branch",
            "V_star":   "2 / (gm/Id)",
        },
        "notes": [
            "V* = 2*Id/gm; matches PDF FoM-peak band (0.15-0.20V) when in moderate inversion"
            if V_star_in and 0.10 < V_star_in < 0.30
            else (f"V*={V_star_in:.3f}V is outside typical FoM-peak band"
                  if V_star_in else "no SR spec given; SR-derived currents skipped"),
        ],
    }


# -----------------------------------------------------------------------------
# Tool 9: check_constraints
# -----------------------------------------------------------------------------
@mcp.tool()
async def check_constraints(metrics: Dict[str, float],
                            spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare measured/simulated metrics against the spec's pass/fail thresholds.

    Spec items can be expressed as:
        "GBW: 10MHz"          → exact match within tolerance
        "DC_gain: '>= 60 dB'" → inequality
        "power: '< 200 uW'"   → inequality
    parse_yaml_spec normalizes numeric values; inequality strings are parsed
    here.

    Args:
        metrics: dict of {name: float} from a simulation (e.g.
                 {"GBW": 9.2e6, "DC_gain": 71.0, "PM": 64.0, "SR": 9.5e6,
                  "power": 76e-6})
        spec: spec dict (must have 'specs' key, as from parse_yaml_spec)

    Returns:
        Dict with:
            - passed (bool): all checks ok
            - per_spec: list of {name, target, measured, passed, margin_pct}
            - violations: list of names that failed
            - notes: human-readable summary lines
    """
    s = spec.get("specs", spec)
    per_spec = []
    violations = []

    for name, target in s.items():
        if name not in metrics:
            per_spec.append({"name": name, "target": target,
                             "measured": None, "passed": None,
                             "note": "metric not provided"})
            continue
        m = float(metrics[name])

        # Inequality string?
        if isinstance(target, str):
            tgt_stripped = target.strip()
            op = None
            if tgt_stripped.startswith(">="): op, num_str = ">=", tgt_stripped[2:]
            elif tgt_stripped.startswith("<="): op, num_str = "<=", tgt_stripped[2:]
            elif tgt_stripped.startswith(">"):  op, num_str = ">",  tgt_stripped[1:]
            elif tgt_stripped.startswith("<"):  op, num_str = "<",  tgt_stripped[1:]
            else:
                per_spec.append({"name": name, "target": target,
                                 "measured": m, "passed": None,
                                 "note": "unrecognized comparator"})
                continue
            # parse the number (could include unit; reuse parse_yaml_spec logic
            # by treating it as a fresh string)
            UNIT_MULT = {"f":1e-15,"p":1e-12,"n":1e-9,"u":1e-6,"m":1e-3,
                         "k":1e3,"M":1e6,"G":1e9}
            # Handle compound units first (longest match wins)
            _compound = False
            for cu, mult in [("V/us", 1e6), ("V/s", 1.0)]:
                if num_str.endswith(cu):
                    num_str_base = num_str[:-len(cu)].strip()
                    try:
                        # Strip a trailing SI prefix if any
                        if num_str_base and num_str_base[-1] in UNIT_MULT:
                            target_val = float(num_str_base[:-1]) * UNIT_MULT[num_str_base[-1]] * mult
                        else:
                            target_val = float(num_str_base) * mult
                        _compound = True
                    except ValueError:
                        target_val = None
                    break
            if _compound:
                pass  # already set
            else:
                for suf in ("Hz","F","V","A","s","W","dB","°"):
                    if num_str.endswith(suf): num_str = num_str[:-len(suf)].strip(); break
                if num_str and num_str[-1] in UNIT_MULT:
                    try:
                        target_val = float(num_str[:-1]) * UNIT_MULT[num_str[-1]]
                    except ValueError:
                        target_val = None
                else:
                    try: target_val = float(num_str)
                    except ValueError: target_val = None
            if target_val is None:
                per_spec.append({"name": name, "target": target,
                                 "measured": m, "passed": None,
                                 "note": "could not parse target number"})
                continue
            passed = {">=": m >= target_val, "<=": m <= target_val,
                      ">":  m >  target_val, "<":  m <  target_val}[op]
            margin = ((m - target_val) / target_val * 100) if target_val else 0.0
            per_spec.append({"name": name, "target": target,
                             "target_val": target_val, "measured": m,
                             "passed": passed, "margin_pct": margin})
            if not passed:
                violations.append(name)
        else:
            # Numeric target — treat as "approximately equal" with 5% tol
            target_val = float(target)
            passed = abs(m - target_val) / max(abs(target_val), 1e-12) < 0.05
            margin = ((m - target_val) / target_val * 100) if target_val else 0.0
            per_spec.append({"name": name, "target": target,
                             "target_val": target_val, "measured": m,
                             "passed": passed, "margin_pct": margin})
            if not passed:
                violations.append(name)

    return {
        "passed": len(violations) == 0,
        "per_spec": per_spec,
        "violations": violations,
        "n_passed": sum(1 for p in per_spec if p.get("passed") is True),
        "n_failed": len(violations),
    }


# -----------------------------------------------------------------------------
# Sizing notebook — Tools 10 & 11
# -----------------------------------------------------------------------------
# Persistent cross-turn state. Stored as JSON file on disk so it survives
# server restarts and is inspectable by the user.

_NOTEBOOK_PATH = Path(os.getenv("ASTRA_NOTEBOOK_DIR", "./notebooks"))
_NOTEBOOK_PATH.mkdir(exist_ok=True)


def _notebook_file(notebook_id: str) -> Path:
    safe_id = "".join(c for c in notebook_id if c.isalnum() or c in "_-")
    return _NOTEBOOK_PATH / f"{safe_id}.json"


def _load_notebook(notebook_id: str) -> Dict[str, Any]:
    fp = _notebook_file(notebook_id)
    if not fp.exists():
        return {"notebook_id": notebook_id, "created_at": datetime.now().isoformat(),
                "entries": []}
    with open(fp) as f:
        return json.load(f)


def _save_notebook(nb: Dict[str, Any]):
    fp = _notebook_file(nb["notebook_id"])
    with open(fp, "w") as f:
        json.dump(nb, f, indent=2, default=str)


@mcp.tool()
async def notebook_append(
        notebook_id: str,
        turn: int,
        decision: str,
        rationale: str = "",
        rejected_options: List[Dict[str, Any]] = None,
        tool_calls: List[Dict[str, Any]] = None,
        metrics: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Append one decision to the sizing notebook for cross-turn memory.

    Call this AFTER every user-confirmed design decision (topology choice,
    L pick, W pick, fix-plan pick, etc.). Skip for pure formula evaluations
    that don't represent a choice.

    The notebook is the LLM's long-term memory across turns — it survives
    context-window compression. Always pair with `notebook_summary()` at the
    start of new decisions to recall prior context.

    Args:
        notebook_id: unique session ID (e.g. "5T_OTA_run01"). Create one per
            sizing session; the user can name it or you can auto-generate.
        turn: turn number in the conversation (1, 2, 3, ...)
        decision: short statement of what was decided
            (e.g. "topology = telescopic_cascode")
        rationale: why this was chosen (1-2 sentences)
        rejected_options: list of {description: str, reason_rejected: str}
            so future turns can avoid revisiting closed branches
        tool_calls: list of {tool: str, args: dict} called during this turn
        metrics: any simulation metrics produced this turn

    Returns:
        Dict with ack + total entries in notebook.
    """
    try:
        nb = _load_notebook(notebook_id)
        nb["entries"].append({
            "turn": turn,
            "timestamp": datetime.now().isoformat(),
            "decision": decision,
            "rationale": rationale,
            "rejected_options": rejected_options or [],
            "tool_calls": tool_calls or [],
            "metrics": metrics or {},
        })
        _save_notebook(nb)
        return {
            "notebook_id": notebook_id,
            "total_entries": len(nb["entries"]),
            "file": str(_notebook_file(notebook_id)),
        }
    except Exception as e:
        return {"error": f"notebook write failed: {e}",
                "traceback": traceback.format_exc()}


@mcp.tool()
async def notebook_summary(
        notebook_id: str,
        last_n_turns: int = None,
) -> Dict[str, Any]:
    """
    Read the sizing notebook for a session. Use this at the START of any
    new design decision to recall what's already been decided and what
    branches were ruled out — prevents re-litigating closed choices.

    Args:
        notebook_id: session ID
        last_n_turns: if set, return only the last N turns (for compact
            context refresh). Default: return everything.

    Returns:
        Dict with:
            - notebook_id, created_at
            - n_entries: total decisions logged
            - locked_decisions: list of {turn, decision} — what's been frozen
            - rejected_branches: list of {decision_context, option, reason}
                across all turns — for the LLM to avoid suggesting again
            - recent_metrics: latest simulation results, if any
    """
    try:
        nb = _load_notebook(notebook_id)
    except Exception as e:
        return {"error": f"could not load notebook: {e}"}

    entries = nb.get("entries", [])
    if last_n_turns is not None:
        entries = entries[-last_n_turns:]

    locked = [{"turn": e["turn"], "decision": e["decision"],
               "rationale": e.get("rationale", "")} for e in entries]

    rejected = []
    for e in entries:
        for opt in e.get("rejected_options", []):
            if isinstance(opt, dict):
                rejected.append({
                    "decision_turn": e["turn"],
                    "decision_context": e["decision"],
                    "option": opt.get("description", str(opt)),
                    "reason": opt.get("reason_rejected", ""),
                })
            else:
                rejected.append({
                    "decision_turn": e["turn"],
                    "decision_context": e["decision"],
                    "option": str(opt),
                    "reason": "",
                })

    recent_metrics = {}
    for e in reversed(entries):
        if e.get("metrics"):
            recent_metrics = e["metrics"]
            break

    return {
        "notebook_id": notebook_id,
        "created_at": nb.get("created_at"),
        "n_entries": len(nb.get("entries", [])),
        "locked_decisions": locked,
        "rejected_branches": rejected,
        "recent_metrics": recent_metrics,
    }


# -----------------------------------------------------------------------------
# Tools 7 & 8 — Simulator wrappers (STUBS pending Garrett's AutoSizer plug-in)
# -----------------------------------------------------------------------------
# These tools will be implemented once Garrett finishes refactoring AutoSizer's
# netlist generator and ngspice runner to be MCP-callable. Stubs are provided
# so Claude Code sees the full tool list and AGENT.md can reference them.

@mcp.tool()
async def generate_netlist(
        yaml_path: str,
        sizing: Dict[str, Any],
) -> Dict[str, Any]:
    """
    [STUB — pending Garrett's AutoSizer integration]

    Generate an ngspice-runnable netlist by filling AutoSizer's template
    with the chosen sizing parameters.

    Args:
        yaml_path: path to the AutoSizer YAML spec (contains topology template)
        sizing: dict mapping device names → {W, L, ...}, e.g.
                {"M1": {"W": 7e-6, "L": 0.7e-6}, ...}

    Returns:
        Dict with netlist_path: filesystem path to the generated .cir file
    """
    return {"error": "generate_netlist not yet plugged in — "
                     "blocked on Garrett's AutoSizer refactor",
            "stub": True,
            "requested_yaml": yaml_path,
            "requested_sizing": sizing}


@mcp.tool()
async def run_single_sim(
        netlist_path: str,
        testbench: List[str] = None,
) -> Dict[str, Any]:
    """
    [STUB — pending Garrett's AutoSizer integration]

    Run one ngspice simulation on a netlist and parse the requested metrics.

    Args:
        netlist_path: path to .cir file (from generate_netlist)
        testbench: list of analyses to run, e.g. ["ac", "dc", "tran"]

    Returns:
        Dict with metrics: {"GBW": ..., "DC_gain": ..., "PM": ..., ...}
    """
    return {"error": "run_single_sim not yet plugged in — "
                     "blocked on Garrett's AutoSizer refactor",
            "stub": True,
            "requested_netlist": netlist_path,
            "requested_testbench": testbench or ["ac", "dc"]}


# -----------------------------------------------------------------------------
# Tool 12: freeze_design_and_export_to_autosizer_yaml
# -----------------------------------------------------------------------------
@mcp.tool()
async def freeze_design_and_export_to_autosizer_yaml(
        notebook_id: str,
        output_path: str,
        sizing: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Finalize an initial design and export it as an AutoSizer-compatible YAML
    seed file for the downstream BO refinement stage.

    Call this once `check_constraints` reports all specs ✓.

    Args:
        notebook_id: the sizing session to freeze
        output_path: where to write the YAML
        sizing: final per-device {W, L, Id} dict

    Returns:
        Dict with path of written YAML and a summary of frozen choices.
    """
    try:
        import yaml
    except ImportError:
        return {"error": "PyYAML not installed; run `pip install pyyaml`"}

    try:
        nb = _load_notebook(notebook_id)
    except Exception as e:
        return {"error": f"could not load notebook: {e}"}

    out_doc = {
        "exported_from": "ASTRA vibe-sizing initial design",
        "notebook_id": notebook_id,
        "exported_at": datetime.now().isoformat(),
        "sizing": sizing,
        "design_history": [
            {"turn": e["turn"], "decision": e["decision"]}
            for e in nb.get("entries", [])
        ],
    }
    try:
        with open(output_path, "w") as f:
            yaml.safe_dump(out_doc, f, sort_keys=False, default_flow_style=False)
    except Exception as e:
        return {"error": f"failed to write YAML: {e}"}

    return {
        "output_path": str(Path(output_path).resolve()),
        "n_decisions_logged": len(nb.get("entries", [])),
        "sizing_summary": sizing,
    }


# Start the server
if __name__ == "__main__":
    # Ensure stdout supports UTF-8
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            print("Stdout encoding set to UTF-8")
        except:
            print("Warning: Could not set stdout encoding to UTF-8")

    print("RAG Server is running, waiting for client connection...")
    mcp.run()