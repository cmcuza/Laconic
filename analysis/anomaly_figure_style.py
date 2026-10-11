"""Fonts and method styles of the anomaly detection figures."""
from __future__ import annotations
import glob
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Locations of the Libertine OpenType files in TeX Live and MacTeX.
LIBERTINE_GLOBS = (
    "/usr/local/texlive/*/texmf-dist/fonts/opentype/public/libertine/LinLibertine_R*.otf",
    "/Library/TeX/Root/texmf-dist/fonts/opentype/public/libertine/LinLibertine_R*.otf",
)


# Legend label, colour and marker of each method, keyed by compressor, in legend order.
METHOD_STYLE = {
    "sz":       ("SZ",      "C1", "s"),
    "mixpiece": ("MXP",     "C2", "D"),
    "serfxor":  ("SXOR",    "C4", "p"),
    "adaedge":  ("ADA",     "C0", "o"),
    "laconic":  ("LACONIC", "C3", "*"),
}


def label(name: str) -> str:
    return METHOD_STYLE[name][0]


def color(name: str) -> str:
    return METHOD_STYLE[name][1]


def marker(name: str) -> str:
    return METHOD_STYLE[name][2]


def use_libertine(mathtext: bool = False) -> None:
    """Register Linux Libertine and make it the serif font, with embedded TrueType glyphs."""
    for pattern in LIBERTINE_GLOBS:
        for path in glob.glob(pattern):
            font_manager.fontManager.addfont(path)
    params = {"font.family": "serif",
              "font.serif": ["Linux Libertine O", "Linux Libertine", "DejaVu Serif"]}
    if mathtext:
        params.update({"mathtext.fontset": "custom",
                       "mathtext.rm": "Linux Libertine O",
                       "mathtext.it": "Linux Libertine O:italic"})
    params.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    plt.rcParams.update(params)
