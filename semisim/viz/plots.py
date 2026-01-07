from pathlib import Path
import numpy as np
from semisim.postprocess.visualization import (
    band_diagram_fig, iv_fig, cv_fig, field_fig, thermal_fig
)

def render_plot(run_dir: Path, what: str, export: Path | None):
    data = np.load(run_dir / "fields.npz")
    if what == "band":
        fig = band_diagram_fig(data["x"], data["Ec"], data["Ev"],
                               data.get("Efn"), data.get("Efp"), data.get("Ef0"))
    elif what == "iv":
        fig = iv_fig(data["V"], data["I"])
    elif what == "cv":
        fig = cv_fig(data["Vg"], data["C"])
    elif what == "fields":
        fig = field_fig(data["x"], data["Efield"])
    elif what == "thermal":
        fig = thermal_fig(data["x"], data["T"])
    else:
        raise ValueError(f"Unknown plot type: {what}")
    if export:
        fig.savefig(export, dpi=180)
    else:
        import matplotlib.pyplot as plt; plt.show()
