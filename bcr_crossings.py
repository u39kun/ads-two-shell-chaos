"""A short BCR-5 trace illustrating the head-on return section.

Save both material shell radii and every crossing around an ordinary two-crossing
return and the first return containing overtaking encounters. Uses the reference solver's dense output in
middle-region Schwarzschild time; no interpolation across crossing updates.

Usage: python bcr_crossings.py results/paper/bcr_crossings.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

import n_shells as N


def crossing_kind(event):
    """Classify from incoming velocities, using the same section as bcr_section."""
    inner, outer = event["v_before"]
    if inner > 0 > outer:
        return "selected" if event["loser"] == "A" else "opposite_head_on"
    if inner * outer > 0:
        return "overtaking"
    raise ValueError("The illustrative trace contains an unclassified crossing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--max-step", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--sample-step", type=float, default=0.01)
    args = parser.parse_args()

    shells = [N.Shell("A", 0.00815, 0.2), N.Shell("B", 0.00815, 0.2)]
    masses = [0.0, 0.015, 0.03]
    state = [
        1.0,
        -np.sqrt(-N.potential(1.0, *masses[:2], shells[0])),
        1.0,
        np.sqrt(-N.potential(1.0, *masses[1:], shells[1])),
    ]
    run = N.simulate(
        shells,
        masses,
        state,
        order=[0, 1],
        tmax=70,
        anchor=1,
        rtol=args.rtol,
        atol=args.rtol * 0.01,
        max_step=args.max_step,
        max_events=1000,
        collect=True,
    )
    if run["status"] != "time_limit":
        raise RuntimeError(f"Trace failed: {run['status']}")
    events = run["events"]
    selected = [i for i, event in enumerate(events) if crossing_kind(event) == "selected"]
    for first, last in zip(selected, selected[1:]):
        kinds = [crossing_kind(event) for event in events[first : last + 1]]
        if "overtaking" in kinds and "opposite_head_on" in kinds:
            break
    else:
        raise RuntimeError("No return containing all three crossing types was found")

    overtaking_pair = first, last
    ordinary_pair = next((a, b) for a, b in zip(selected, selected[1:]) if b - a == 2)
    examples = []
    for name, (first, last) in (("ordinary", ordinary_pair), ("overtaking", overtaking_pair)):
        t0, t1 = events[first]["t"], events[last]["t"]
        padding = min(0.7, 0.07 * (t1 - t0))
        lo, hi = t0 - padding, t1 + padding
        segments = []
        for flight in run["segments"]:
            a, b = max(lo, flight["t"][0]), min(hi, flight["t"][-1])
            if a >= b:
                continue
            times = np.linspace(a, b, max(3, int(np.ceil((b - a) / args.sample_step)) + 1))
            states = flight["dense"](times)
            segments.append(dict(t=times.tolist(), R_A=states[0].tolist(), R_B=states[2].tolist()))

        shown = []
        for event in events:
            if lo <= event["t"] <= hi:
                order = [0, 1] if event["loser"] == "A" else [1, 0]
                factors, _ = N.clock_factors(event["state_before"], event["masses_before"], order, shells, anchor=1)
                shown.append(
                    dict(
                        t=event["t"],
                        R=event["r"],
                        kind=crossing_kind(event),
                        number=event["number"],
                        pair=event["pair"],
                        state_before=event["state_before"],
                        masses_before=event["masses_before"],
                        incoming_dR_dt=[event["state_before"][1] * factors[0], event["state_before"][3] * factors[1]],
                    )
                )
        if sum(event["kind"] == "selected" for event in shown) != 2:
            raise RuntimeError("Each plotting window must contain exactly two selected crossings")
        examples.append(
            dict(
                name=name,
                window=[lo, hi],
                return_times=[t0, t1],
                crossings_per_return=last - first,
                events=shown,
                segments=segments,
            )
        )
    result = dict(
        parameters=run["parameters"],
        solver="reference DOP853, dense output",
        status=run["status"],
        max_constraint_residual=run["max_constraint_residual"],
        sample_step=args.sample_step,
        examples=examples,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result) + "\n")
    print(f"wrote {args.out}")
    for example in examples:
        t0, t1 = example["return_times"]
        print(f"{example['name']}: return {t0:.9f} to {t1:.9f}, {example['crossings_per_return']} crossings")
        for event in example["events"]:
            print(
                f"  t-t0={event['t'] - t0: .6f}, R={event['R']:.6f}, "
                f"{event['kind']}, incoming pair={event['pair']}, dR/dt={event['incoming_dR_dt']}"
            )


if __name__ == "__main__":
    main()
