"""Shared visual style and paths; contains no training or SHAP generation."""
from pathlib import Path
from hashlib import sha256

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = PROJECT_ROOT
OUTPUT_ROOT = PAPER_ROOT / "appendix_e"
DATA_DIR = OUTPUT_ROOT / "data"
FIGURE_DIR = OUTPUT_ROOT / "figures"
NOTEBOOK_DIR = OUTPUT_ROOT / "notebooks"
METHOD_ORDER = ["random", "kmeans", "pred_stratified", "cte"]
METHOD_LABELS = {"random": "Random", "kmeans": "K-Means", "pred_stratified": "PredStrat", "cte": "CTE"}
METHOD_COLORS = {"random": "#4393C3", "kmeans": "#D6604D", "pred_stratified": "#56A65A", "cte": "#9970AB"}
METHOD_MARKERS = {"random": "o", "kmeans": "s", "pred_stratified": "^", "cte": "D"}
DATASET_ORDER = ["german", "diabetes", "acs", "gmsc"]
DATASET_LABELS = {"german": "German Credit", "diabetes": "Diabetes", "acs": "ACS Income", "gmsc": "GMSC", "arrhythmia": "Arrhythmia"}
MODEL_LABELS = {"mlp": "MLP", "ftt": "FTT"}
HEATMAP_CMAP = "YlOrRd"


def set_plot_style():
    """Match the serif type and established method colors of the paper."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11, "axes.labelsize": 11,
        "axes.titlesize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "axes.linewidth": .7, "lines.linewidth": 1.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "pdf.fonttype": 42, "ps.fonttype": 42, "text.usetex": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 180,
    })


def ensure_directories():
    for path in (DATA_DIR, FIGURE_DIR, NOTEBOOK_DIR):
        path.mkdir(parents=True, exist_ok=True)


def save_figure(fig, stem):
    """Save vector PDF and review PNG without closing the notebook figure."""
    ensure_directories()
    path = Path(stem)
    if not path.is_absolute():
        path = FIGURE_DIR / path
    path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf, png = path.with_suffix(".pdf"), path.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=.08,
                metadata={"CreationDate": None, "ModDate": None, "Creator": "Matplotlib"})
    fig.savefig(png, bbox_inches="tight", pad_inches=.08, dpi=180)
    return {"pdf": str(pdf), "png": str(png)}


def file_sha256(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
