# Interactive demo

`../docs/demo/index.html` animates seven orbits:

- The reconstructed BCR-5 example.
- An orbit in the chaotic saddle layer.
- An orbit outside the saddle layer.
- A loop around an island.
- An elliptic fixed point.
- A chaotic orbit visiting the section's two pointed ends.
- BCR-T2 data at δ = 0.03.

GitHub Pages serves the demo at https://u39kun.github.io/ads-two-shell-chaos/demo/ (`../docs/index.html` redirects there).

To rebuild it from the code in this repository:

```
# from the repository root
python demo/gen_data.py . demo/orbits.json
python demo/build.py docs/demo/index.html --standalone
```
