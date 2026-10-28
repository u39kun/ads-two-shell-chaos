"""Shared setup for the scripts: the reference solver n_shells.py, whose simulate() is replaced by the
compiled drop-in fastshells.simulate unless SOLVER=reference, and the two identical shells of BCR's Fig. 5 example
(w = 0.2, m = 0.9*20^(-0.4)*0.03, total mass 0.03)."""

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
