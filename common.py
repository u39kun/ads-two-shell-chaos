"""Shared setup for the scripts: the reference solver n_shells.py, whose simulate() is replaced by the
compiled drop-in fastshells.simulate unless SOLVER=reference, and two identical shells A, B with the formula mass
m = 0.9*20^(-0.4)*0.03, w = 0.2, total mass 0.03 (used by two_shell_measure.py; the BCR-5 scripts set BCR's printed
m = 0.00815 themselves)."""

import sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import n_shells as N

if os.environ.get("SOLVER", "fast") == "fast":
    import fastshells

    N.simulate = fastshells.simulate  # compiled drop-in; SOLVER=reference keeps the scipy solver of n_shells.py

TOTAL = 0.03
M_A = 0.9 * 20 ** (-0.4) * TOTAL
A, B = N.Shell("A", M_A), N.Shell("B", M_A)
