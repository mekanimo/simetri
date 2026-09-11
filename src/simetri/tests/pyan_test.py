from pathlib import Path

import pyan

# from IPython.display import HTML

ROOT = Path(r"c:/uv_simetri_3.9/simetri/src")  # package root sits here
FILES = str(ROOT / "simetri" / "**" / "*.py")

html = pyan.create_callgraph(
    filenames=FILES,
    root=str(ROOT),
    namespace="simetri.geom",
    format="html",
    draw_defines=False,
    draw_uses=True,
    exclude=["*/gui/*", "*/gui/**"],  # one of these is enough; try */gui/*
)
# HTML(html)
Path("callgraph.html").write_text(html, encoding="utf-8")
# pyan3 c:/uv_simetri_3.9/simetri/src/simetri/geom/*.py --uses --no-defines --colored --grouped --html --root c:/uv_simetri_3.9/simetri/src > geom_uses.html
