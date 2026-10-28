# Interactive demo

`../docs/demo/index.html` animates the two shells for seven orbits at the parameters of BCR's example: BCR's example itself, an orbit in the chaotic layer, one just outside it, a loop around an island, the elliptic fixed point, a chaotic orbit near the map's two pointed ends, and BCR's type (ii) data at δ = 0.03. Each orbit has a twin nudged by 5×10⁻⁵, BCR's nudge. GitHub Pages serves it at https://u39kun.github.io/ads-two-shell-chaos/demo/ (`../docs/index.html` redirects there).

To rebuild it from the code in this repository (a few minutes):

```
# from repo root
python demo/gen_data.py . demo/orbits.json
python demo/build.py docs/demo/index.html --standalone
```

- `gen_data.py` computes each orbit and its twin with the paper's solver and stores the state after every crossing. The two orbits from the paper's Fig. 3 take their section points from `tangent_map.py`, called exactly as `bcr_layer_tangent.py` calls it, so they follow the paper's trajectories. The orbit near the ends starts at the point of the paper's Fig. 7 orbit (`bcr_section_overview.py`) with the same solver settings, and the edge of the map is read from that script's results.
- `physics.js` integrates the motion between crossings in the browser, restarting from the stored state at each crossing.
- `template.html` is the page; `build.py` inlines the data and the integrator into it.
