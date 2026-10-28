"""Addendum: BCR's nearby-orbit test (Fig. 4, Sec. 3.3) with looser tolerances and a larger maximum step.

Runs ../bcr_lyapunov.py unchanged, except that every integration uses relative tolerance 1e-8, absolute tolerance
1e-10 and a maximum step of 0.5, instead of the reference 1e-12, 1e-14 and 0.02. The step cap is loosened too because at
the reference cap the step size, not the tolerance, controls the error. The paper (App. B, "Nearby-orbit checks",
cited in Sec. 3.3 and the Discussion) quotes this run for the initially separating assignment (--start D), to
t = 2000: its envelope values, threshold times, slopes and fitted rates equal those of
../results/paper/bcr_lyapunov.json at the precision quoted.
All other arguments are passed through to bcr_lyapunov.py. The settings are added to the output's params.
usage: python addendum/nudge_lowprec.py OUT.json --tmax 2000 [other bcr_lyapunov.py options]"""

import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bcr_lyapunov as BL  # sets SOLVER=reference, as for the shipped nudge results

LOW = dict(rtol=1e-8, atol=1e-10, max_step=0.5)
_simulate = BL.N.simulate
_trajectory = BL.trajectory


def _simulate_low(*args, **kwargs):
    kwargs.update(LOW)
    return _simulate(*args, **kwargs)


def trajectory(args):
    # patched here, in the worker, so the override holds for any multiprocessing start method
    BL.N.simulate = _simulate_low
    return _trajectory(args)


if __name__ == "__main__":
    BL.trajectory = trajectory  # bcr_lyapunov.main() maps the module-level name over its runs
    print(f"integration settings: {LOW}")
    BL.main()
    out = sys.argv[1]
    data = json.load(open(out))
    data["params"]["integration"] = LOW
    json.dump(data, open(out, "w"))
