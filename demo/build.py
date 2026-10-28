"""Build the trajectory explorer page from template.html, physics.js, and orbits.json (made by gen_data.py).
python build.py OUT.html                  page body for a host that supplies <html>/<head> 
python build.py OUT.html --standalone     complete HTML document, e.g. docs/demo/index.html for GitHub Pages"""

import sys, os, urllib.parse

here = os.path.dirname(os.path.abspath(__file__))
t = open(os.path.join(here, "template.html")).read()
data = open(os.path.join(here, "orbits.json")).read()
assert "</" not in data
t = t.replace("/*DATA*/", data).replace("/*PHYSICS*/", open(os.path.join(here, "physics.js")).read())
if "--standalone" in sys.argv:
    i = t.index('<div class="wrap">')
    head, body = t[:i], t[i:]
    t = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        # favicon: the two shells as concentric rings, A orange inside B teal, in the page's dark-theme colors when the
        # browser is dark
        '<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,'
        + urllib.parse.quote(
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><style>.b{stroke:#11746a}.a{stroke:#c0501a}"
            "@media (prefers-color-scheme: dark){.b{stroke:#3dc0b0}.a{stroke:#f0884a}}</style>"
            "<circle class='b' cx='16' cy='16' r='13' fill='none' stroke-width='4'/>"
            "<circle class='a' cx='16' cy='16' r='6.5' fill='none' stroke-width='4'/></svg>",
            safe="/:=' ",
        )
        + '">\n'
        '<meta name="description" content="Animated orbits of two thin shells in anti-de Sitter space, with their '
        'Poincaré map and a nudged twin that shows regular versus chaotic separation.">\n'
        "<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}"
        "body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n"
        + head
        + "</head>\n<body>\n"
        + body
        + "</body>\n</html>\n"
    )
out = [a for a in sys.argv[1:] if not a.startswith("--")][0]
open(out, "w").write(t)
print("wrote", out, os.path.getsize(out) // 1024, "KB")
