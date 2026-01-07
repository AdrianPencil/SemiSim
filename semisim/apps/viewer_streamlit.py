import streamlit as st, numpy as np
from pathlib import Path
from semisim.postprocess.visualization import band_diagram_fig, iv_fig, cv_fig, field_fig, thermal_fig

st.set_page_config(page_title="SemiSim Viewer", layout="wide")
run_dir = Path(st.sidebar.text_input("Run directory", "runs/moscap"))
what = st.sidebar.selectbox("What to view", ["band","iv","cv","fields","thermal"])

if not (run_dir / "fields.npz").exists():
    st.warning("No fields.npz found in run directory"); st.stop()

data = np.load(run_dir / "fields.npz")
st.title("SemiSim Viewer")

if what == "band":
    fig = band_diagram_fig(data["x"], data["Ec"], data["Ev"],
                           data.get("Efn"), data.get("Efp"), data.get("Ef0"), annotate=True)
elif what == "iv":
    fig = iv_fig(data["V"], data["I"])
elif what == "cv":
    fig = cv_fig(data["Vg"], data["C"])
elif what == "fields":
    fig = field_fig(data["x"], data["Efield"])
else:
    fig = thermal_fig(data["x"], data["T"])
st.pyplot(fig)
st.caption("Educational mode: overlays theory lines and labels regions for quick intuition.")
