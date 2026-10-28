"""Long two-orbit exponent for BCR's Fig. 5 example (Fig. 2c of the paper): 2e4 returns at d0 = 1e-10 and 1e-9.
Reads results/bcr_section/bcr_section.json and writes results/bcr_section/bcr_benettin_long.json.
usage: python bcr_benettin_long.py"""

import sys, json, os, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from multiprocessing import Pool
import bcr_section_checks as C

S = json.load(open(os.path.join(HERE, "results", "bcr_section", "bcr_section.json")))
bcr0 = S["bcr_orbit"][0]["points"][0]
tasks = [("BCR", bcr0, 1e-10, 1e-12, 20000, 10), ("BCR", bcr0, 1e-9, 1e-12, 20000, 10)]


def run(t):
    r = C.benettin(t)
    print(r["name"], r["d0"], [(k, f"{l:.2e}") for k, l in r["lambda_N"]], flush=True)
    return r


if __name__ == "__main__":
    with Pool(2) as p:
        res = p.map(run, tasks)
    json.dump(res, open(os.path.join(HERE, "results", "bcr_section", "bcr_benettin_long.json"), "w"))
    print("done", flush=True)
