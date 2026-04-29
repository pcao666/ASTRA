"""
ngspice_runner.py
=================
Drop-in replacement for `lyngspice.NgSpice` used by ASTRA's simulation_OTA_two.py.

Original interface:
    ng = NgSpice()
    data, units = ng.run(netlist_path)
    # data['ac1']['net4']        -> complex numpy array (AC voltage at net4)
    # data['ac1']['frequency']   -> frequency points
    # data['op1']['v0#branch']   -> DC current through V0

This module replicates that interface using subprocess + ngspice's `wrdata` output,
removing the dependency on the non-public KATO/lyngspice package.

Strategy:
1. Parse the user's netlist to find AC analysis nodes referenced (net3, net4, etc.)
2. Inject a `.control` block that runs the simulation and uses `wrdata` to dump
   AC and OP results to text files.
3. Run ngspice in batch mode: subprocess.run(["ngspice", "-b", netlist])
4. Parse the wrdata output files into the same dict-of-arrays structure.

Author: Penny's team, 2026
"""

import os
import subprocess
import tempfile
import shutil
import re
import numpy as np


# Nets that the original code accesses in AC analysis.
# These match the conventions in simulation_OTA_two.py:
#   data['ac1']['net3']  -> input AC voltage
#   data['ac1']['net4']  -> output AC voltage
# Add more here if your netlist uses different node names.
AC_NETS = ['net3', 'net4', 'vout', 'inp']

# Voltage sources whose branch currents we want to capture for OP analysis.
# These are matched against actual V* sources in the netlist (case-insensitive).
OP_BRANCHES_VSOURCES = ['v0']


class NgSpice:
    """Drop-in replacement for lyngspice.NgSpice.

    Usage:
        ng = NgSpice()
        data, units = ng.run("/path/to/netlist.cir")
        # data['ac1']['net4']         -> complex numpy array
        # data['ac1']['frequency']    -> float numpy array (Hz)
        # data['op1']['v0#branch']    -> 1-element array (DC current)
    """

    def __init__(self, ngspice_bin='ngspice', timeout=120, debug=False):
        self.ngspice_bin = ngspice_bin
        self.timeout = timeout
        self.debug = debug

    def run(self, netlist_path):
        """Run a netlist file. Returns (data_dict, units_dict).

        data_dict structure mirrors lyngspice:
            data['ac1'][node_name]    -> np.ndarray (complex for voltages)
            data['ac1']['frequency']  -> np.ndarray (float)
            data['op1'][branch]       -> np.ndarray (1 element)

        units_dict is provided for API compatibility; values are placeholders.
        """
        if not os.path.exists(netlist_path):
            raise FileNotFoundError(f"Netlist not found: {netlist_path}")

        # Create a temp working directory so multiple runs don't collide
        work_dir = tempfile.mkdtemp(prefix='ngspice_run_')
        try:
            # Read original netlist
            with open(netlist_path, 'r') as f:
                netlist_text = f.read()

            # Inject a measurement-friendly .control block before .end
            instrumented_netlist = self._instrument_netlist(netlist_text, work_dir)

            instrumented_path = os.path.join(work_dir, 'sim.cir')
            with open(instrumented_path, 'w') as f:
                f.write(instrumented_netlist)

            # Run ngspice
            result = subprocess.run(
                [self.ngspice_bin, '-b', '-o', os.path.join(work_dir, 'ngspice.log'),
                 instrumented_path],
                cwd=work_dir,
                capture_output=True, text=True, timeout=self.timeout
            )

            if self.debug:
                print(f"[ngspice stdout]\n{result.stdout}")
                print(f"[ngspice stderr]\n{result.stderr}")

            # Parse outputs
            data = self._parse_outputs(work_dir)

            # Provide units dict for compat (mostly empty)
            units = self._build_units_stub(data)

            return data, units

        finally:
            if not self.debug:
                shutil.rmtree(work_dir, ignore_errors=True)
            else:
                print(f"[debug] preserved work dir: {work_dir}")

    # ------------------------------------------------------------------
    # Netlist instrumentation
    # ------------------------------------------------------------------
    def _instrument_netlist(self, netlist_text, work_dir):
        """Add a .control block that runs AC and OP analyses separately,
        dumping each set of results before the next analysis overwrites them.

        Strategy:
          1. Strip any existing .ac and .op statements from the netlist body
             (they will be invoked from .control with `ac ...` and `op` commands).
          2. Strip any existing .control block (we replace it with our own).
          3. Build a .control that:
                - runs `ac dec ...` -> wrdata each AC node
                - runs `op`         -> wrdata each branch current
        """
        # --- Discover nodes and Vsources ---
        present_nodes = self._extract_nodes(netlist_text)
        nodes_to_dump = [n for n in AC_NETS if n in present_nodes]

        vsources = self._extract_vsources(netlist_text)
        op_vsources = [v for v in OP_BRANCHES_VSOURCES if v in vsources]
        if not op_vsources and 'v0' in vsources:
            op_vsources = ['v0']

        # --- Extract AC analysis parameters from .ac line ---
        ac_match = re.search(r'^\.ac\s+(.+?)$', netlist_text,
                             re.IGNORECASE | re.MULTILINE)
        ac_args = ac_match.group(1).strip() if ac_match else None

        # --- Strip existing .ac, .op, .control blocks from netlist body ---
        body = netlist_text
        body = re.sub(r'^\.ac\s+.+?$', '', body, flags=re.IGNORECASE | re.MULTILINE)
        body = re.sub(r'^\.op\s*$', '', body, flags=re.IGNORECASE | re.MULTILINE)
        body = re.sub(r'\.control\s*\n.*?\.endc\s*\n', '', body,
                       flags=re.IGNORECASE | re.DOTALL)

        # --- Build new .control block ---
        ctrl = ['.control']

        if ac_args and nodes_to_dump:
            ctrl.append(f'ac {ac_args}')
            for node in nodes_to_dump:
                ctrl.append(
                    f"wrdata {os.path.join(work_dir, f'ac_{node}.txt')} v({node})"
                )

        if op_vsources:
            ctrl.append('op')
            for vsrc in op_vsources:
                ctrl.append(
                    f"wrdata {os.path.join(work_dir, f'op_{vsrc}.txt')} i({vsrc})"
                )

        ctrl.append('.endc')
        ctrl_block = '\n'.join(ctrl) + '\n'

        # --- Insert .control block before .end ---
        end_match = re.search(r'^\.end\s*$', body, re.MULTILINE)
        if end_match:
            new_text = (
                body[:end_match.start()] +
                ctrl_block + '\n' +
                body[end_match.start():]
            )
        else:
            new_text = body + '\n' + ctrl_block + '\n.end\n'

        return new_text

    def _extract_nodes(self, netlist_text):
        """Extract the set of nodes referenced by component lines.

        Component lines start with R, L, C, V, I, M, X, etc. and list nodes
        as the 2nd...Nth tokens. We collect any token that looks like a node
        name (alphanumeric + underscore, not a parameter).
        """
        nodes = set()
        for line in netlist_text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith('*') or stripped.startswith('.'):
                continue
            tokens = stripped.split()
            if not tokens:
                continue
            # Component lines: first letter denotes element type
            first = tokens[0][0].upper()
            if first in 'RLCVIMXBDQJ':
                # Skip the device name (token 0); collect node-like tokens
                # until we hit a model name or parameter (containing '=')
                for tok in tokens[1:]:
                    if '=' in tok:
                        break
                    # Node names are typically alphanumeric/underscore
                    if re.match(r'^[a-zA-Z0-9_]+$', tok):
                        nodes.add(tok.lower())
        return nodes

    def _extract_vsources(self, netlist_text):
        """Extract names of all voltage sources (V*) declared in the netlist."""
        vsources = set()
        for line in netlist_text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith('*') or stripped.startswith('.'):
                continue
            tokens = stripped.split()
            if tokens and tokens[0][0].upper() == 'V':
                vsources.add(tokens[0].lower())
        return vsources

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------
    def _parse_outputs(self, work_dir):
        """Read all wrdata output files and assemble the data dict."""
        data = {'ac1': {}, 'op1': {}}

        # --- Parse AC files ---
        # wrdata format for AC: index  freq  re(v(node))  im(v(node))
        # Each file has 4 columns when written with `wrdata file v(node)`:
        #    index  freq  re  im   (but actually: freq, re_val, freq, im_val per pair)
        # ngspice wrdata writes pairs: <xvar> <yvar_re> <xvar> <yvar_im>
        # for complex data. Let's parse robustly.
        freq_array = None
        for fname in os.listdir(work_dir):
            if not fname.startswith('ac_') or not fname.endswith('.txt'):
                continue
            node_name = fname[3:-4]  # strip 'ac_' and '.txt'
            filepath = os.path.join(work_dir, fname)

            freq_vec, complex_vec = self._parse_ac_file(filepath)
            if freq_vec is None:
                continue

            data['ac1'][node_name] = complex_vec
            if freq_array is None:
                freq_array = freq_vec

        if freq_array is not None:
            data['ac1']['frequency'] = freq_array

        # --- Parse OP files ---
        # wrdata for DC OP: <xvar> <yval> [<xvar> <yval_im>...]
        # Since OP is a single point, often we get a single line.
        # However in batch mode, wrdata may not write OP results unless we use
        # `op` followed by `wrdata`. We do this in the injection.
        for fname in os.listdir(work_dir):
            if not fname.startswith('op_') or not fname.endswith('.txt'):
                continue
            vsrc = fname[3:-4]
            filepath = os.path.join(work_dir, fname)
            current = self._parse_op_file(filepath)
            if current is not None:
                # Original code uses 'v0#branch' as key; preserve that
                data['op1'][f'{vsrc}#branch'] = current

        return data

    def _parse_ac_file(self, filepath):
        """Parse an AC wrdata file written by `wrdata file v(node)`.

        Format observed in ngspice 42 batch mode:
            col 0: frequency
            col 1: re(v(node))
            col 2: im(v(node))

        Older lyngspice docs claim a 4-column format (freq, re, freq, im) for
        complex variables, but ngspice 42 produces 3 columns when only one
        variable is requested. We handle both robustly.
        """
        try:
            arr = np.loadtxt(filepath)
        except (ValueError, OSError):
            return None, None

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        ncols = arr.shape[1]
        if ncols == 3:
            # freq, re, im
            freq = arr[:, 0]
            complex_vec = arr[:, 1] + 1j * arr[:, 2]
        elif ncols >= 4:
            # freq, re, freq_dup, im (older format)
            freq = arr[:, 0]
            complex_vec = arr[:, 1] + 1j * arr[:, 3]
        else:
            return None, None

        return freq, complex_vec

    def _parse_op_file(self, filepath):
        """Parse a DC operating point wrdata file."""
        try:
            arr = np.loadtxt(filepath)
        except (ValueError, OSError):
            return None

        if arr.ndim == 0:
            return np.array([float(arr)])
        if arr.ndim == 1:
            # Format: <xvar> <yval>  -> take 2nd column
            if len(arr) >= 2:
                return np.array([arr[1]])
            return np.array([arr[0]])
        # 2D: take last row, 2nd column (DC OP is a single point but ngspice
        # may emit multiple lines if AC sweep ran; the OP is from .op which
        # should be a single point)
        if arr.shape[1] >= 2:
            return np.array([arr[-1, 1]])
        return np.array([arr[-1, 0]])

    def _build_units_stub(self, data):
        """Construct a stub units dict for API compatibility."""
        units = {'ac1': {}, 'op1': {}}
        for analysis in data:
            for key in data[analysis]:
                if key == 'frequency':
                    units[analysis][key] = 'Hz'
                elif analysis == 'ac1':
                    units[analysis][key] = 'V'
                elif analysis == 'op1':
                    units[analysis][key] = 'A'
        return units


# Convenience function for quick testing
def run_netlist(netlist_path, debug=False):
    """One-shot wrapper: run a netlist and return (data, units)."""
    return NgSpice(debug=debug).run(netlist_path)


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python ngspice_runner.py <netlist.cir>")
        sys.exit(1)

    data, units = run_netlist(sys.argv[1], debug=True)
    print("\n=== AC analysis ===")
    for k, v in data['ac1'].items():
        if hasattr(v, '__len__'):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")
    print("\n=== OP analysis ===")
    for k, v in data['op1'].items():
        print(f"  {k}: {v}")
