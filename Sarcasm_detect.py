# ---------- Imports ----------
import json
import re
from email.policy import default
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, precision_recall_fscore_support
)

from PIL import Image  # <-- needed for PDF image embedding

def _go(page_name: str):
    st.session_state["selected_page"] = page_name
    # works in current and older Streamlit
    try:
        st.rerun()
    except Exception:
        st.rerun()

st.set_page_config(page_title="Sarcasm & Toxicity Detector", layout="wide")
# ---------- Session state defaults (avoid KeyError) ----------
for k, v in {
    "featurizer_selected": None,
    "featurizer": None,
    "predict_featurizer": None,
    "trained_featurizer": None,
    "tfidf": None,
    "glove": None,
    "glove_dim": None,
    "glove_scaler": None,
    "lr_model": None,
    "dt_model": None,
    "threshold": 0.5,
    "active_model_name": "Logistic Regression",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- Setup ----------
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

# ---------- Cleaning ----------
def clean_text(s) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(toks)

# ---------- Data (fixed path) ----------
RAW_PATH = Path("C:/Users/kojon/PycharmProjects/Text Analytics Projects/Text Anaytics/Sarcasm_Headlines_Dataset.json")

def parse_jsonl_path(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

if not RAW_PATH.exists():
    st.error(f"Dataset file not found: {RAW_PATH}")
    st.stop()

data_rows = parse_jsonl_path(RAW_PATH)
s_df = pd.DataFrame(data_rows)
if not {"headline", "is_sarcastic"}.issubset(s_df.columns):
    st.error("Dataset must include 'headline' and 'is_sarcastic' columns.")
    st.stop()

# ---------- Sidebar: featurizer choice ----------
st.sidebar.header("🧠 Featurization")
vec_choice = st.sidebar.radio("Choose features", ["TF-IDF", "GloVe (average)"], index=0)

# Reset models if featurizer changed
def reset_models_on_featurizer_change(choice: str):
    prev = st.session_state.get("featurizer_selected")
    if prev != choice:
        for k in [
            "lr_model", "dt_model",
            "tfidf",
            "glove", "glove_dim", "glove_scaler",
            "predict_featurizer", "trained_featurizer",
        ]:
            st.session_state[k] = None
        st.session_state["featurizer_selected"] = choice

reset_models_on_featurizer_change(vec_choice)

# ---------- GloVe settings ----------
GLOVE_PATH = Path("C:/Users/kojon/PycharmProjects/Text Analytics Projects/Text Anaytics/glove.6B.100d.txt")
GLOVE_DIM = 100   # must match your file (50/100/200/300)
USE_SCALER = True

# ---------- GloVe utilities ----------
@st.cache_resource(show_spinner=False)
def load_glove(path: Path, dim: int):
    if not path.exists():
        raise FileNotFoundError(f"GloVe file not found: {path}")
    emb = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            if vec.size == dim:
                emb[word] = vec
    if not emb:
        raise ValueError(f"No vectors loaded. Check that dim={dim} matches your file.")
    return emb

def text_to_glove_vec(text: str, emb: dict, dim: int) -> np.ndarray:
    toks = clean_text(text).split()
    if not toks:
        return np.zeros(dim, dtype=np.float32)
    vecs = [emb[t] for t in toks if t in emb]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0)

def transform_texts_to_glove(texts, emb: dict, dim: int) -> np.ndarray:
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        out[i] = text_to_glove_vec(t, emb, dim)
    return out

# ---------- Preprocess ----------
s_df["is_sarcastic"] = pd.to_numeric(s_df["is_sarcastic"], errors="coerce").fillna(0).astype(int)
s_df["clean"] = s_df["headline"].astype(str).apply(clean_text)
y = s_df["is_sarcastic"].astype(int)

X_train_txt, X_test_txt, y_train, y_test = train_test_split(
    s_df["clean"], y, test_size=0.2, random_state=42, stratify=y
)

# Build features based on choice
if vec_choice == "TF-IDF":
    tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2)
    X_train_vec = tfidf.fit_transform(X_train_txt)
    X_test_vec  = tfidf.transform(X_test_txt)
    st.session_state["featurizer"] = "tfidf"
    st.session_state["tfidf"] = tfidf
else:
    try:
        glove = load_glove(GLOVE_PATH, GLOVE_DIM)
    except Exception as e:
        st.error(str(e))
        st.stop()
    X_train_vec = transform_texts_to_glove(X_train_txt.values, glove, GLOVE_DIM)
    X_test_vec  = transform_texts_to_glove(X_test_txt.values, glove, GLOVE_DIM)
    scaler = None
    if USE_SCALER:
        scaler = StandardScaler()
        X_train_vec = scaler.fit_transform(X_train_vec)
        X_test_vec  = scaler.transform(X_test_vec)
    st.session_state["featurizer"] = "glove"
    st.session_state["glove"] = glove
    st.session_state["glove_dim"] = GLOVE_DIM
    st.session_state["glove_scaler"] = scaler

# ---------- Models ----------
dt_model = DecisionTreeClassifier(max_depth=None, min_samples_split=4, random_state=42)
lr_model = LogisticRegression(max_iter=2000, solver="liblinear", random_state=42)

# ---------- Evaluation helper ----------
def evaluate_model(name, model, Xtr, ytr, Xte, yte, thr=0.5):
    model.fit(Xtr, ytr)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(Xte)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(Xte)
    else:
        y_score = model.predict(Xte).astype(float)

    if hasattr(model, "predict_proba") or hasattr(model, "decision_function"):
        y_pred = (y_score >= thr).astype(int)
    else:
        y_pred = model.predict(Xte)

    acc = accuracy_score(yte, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(yte, y_score)
    except Exception:
        auc = np.nan
    cm = confusion_matrix(yte, y_pred)
    return {"name": name, "model": model, "y_score": y_score, "cm": cm,
            "acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}

# ---------- Report helpers ----------
def _draw_confusion_matrix(c, cm, x, y, title="Confusion Matrix", cell_w=70, cell_h=24):
    # Draw a 2x2 CM on reportlab canvas `c` at top-left (x,y)
    from reportlab.lib import colors
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y + cell_h * 3 + 8, title)
    c.setFont("Helvetica", 9)
    c.drawString(x + cell_w*1.2, y + cell_h*2.5 + 2, "Predicted")
    c.rotate(90); c.drawString(y - 10 + cell_h*0.7, -x + 8, "Actual"); c.rotate(-90)
    c.setStrokeColor(colors.black)
    for i in range(3): c.line(x + i*cell_w, y, x + i*cell_w, y + 2*cell_h)
    for j in range(3): c.line(x, y + j*cell_h, x + 2*cell_w, y + j*cell_h)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x + 4, y + cell_h*2 + 6, "0"); c.drawString(x + cell_w + 4, y + cell_h*2 + 6, "1")
    c.drawString(x - 12, y + cell_h + 6, "0"); c.drawString(x - 12, y + 6, "1")
    c.setFont("Helvetica", 11)
    c.drawCentredString(x + cell_w/2, y + cell_h*1.5, str(int(cm[0,0])))
    c.drawCentredString(x + cell_w*1.5, y + cell_h*1.5, str(int(cm[0,1])))
    c.drawCentredString(x + cell_w/2, y + cell_h*0.5, str(int(cm[1,0])))
    c.drawCentredString(x + cell_w*1.5, y + cell_h*0.5, str(int(cm[1,1])))

def _fig_to_pil(fig) -> Image.Image:
    """Convert a Matplotlib figure to a Pillow Image."""
    buf = BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight", dpi=180)
    buf.seek(0)
    return Image.open(buf)

def generate_pdf_report_enhanced(dataset_info: dict, metrics_df: pd.DataFrame, cms: dict, roc_fig, pr_fig) -> bytes | None:
    """
    Polished PDF with dataset summary, metrics, confusion matrices, ROC & PR curves.
    Returns BYTES (not a BytesIO), so st.download_button works cleanly.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except Exception as e:
        st.error(
            "Report generation needs the 'reportlab' package.\n"
            "Install with:\n\npip install reportlab pillow\n\n"
            f"Details: {e}"
        )
        return None

    # Convert figures to Pillow images (avoid ImageReader.format error)
    roc_img = _fig_to_pil(roc_fig)
    pr_img  = _fig_to_pil(pr_fig)

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    # ---- Cover/Summary ----
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Sarcasm Detection Report")

    c.setFont("Helvetica", 11)
    y = height - 80
    def _line(text):
        nonlocal y
        c.drawString(50, y, text); y -= 16

    _line(f"Samples (total): {dataset_info.get('n_rows')}")
    _line(f"Class balance (0/1): {dataset_info.get('counts_str')}")
    _line(f"Featurizer: {dataset_info.get('featurizer')}  |  Feature dim: {dataset_info.get('feat_dim')}")
    _line(f"Decision threshold: {dataset_info.get('threshold'):.2f}")

    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Model Evaluation Metrics")
    y -= 18
    c.setFont("Helvetica", 10)

    for model, row in metrics_df.iterrows():
        c.drawString(
            58, y,
            f"{model}:  Acc={row['Accuracy']:.3f}  Prec={row['Precision']:.3f}  "
            f"Rec={row['Recall']:.3f}  F1={row['F1']:.3f}  AUC={row['ROC-AUC']:.3f}"
        )
        y -= 14

    y -= 6
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, y, "Notes: Precision = TP/(TP+FP) | Recall = TP/(TP+FN) | F1 = harmonic mean of precision & recall")
    c.showPage()

    # ---- Confusion Matrices ----
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 40, "Confusion Matrices")
    top_y = height - 120
    left_x = 70; right_x = 330
    if "Logistic Regression" in cms and cms["Logistic Regression"] is not None:
        _draw_confusion_matrix(c, np.array(cms["Logistic Regression"]), left_x, top_y - 60, title="Logistic Regression")
    if "Decision Tree" in cms and cms["Decision Tree"] is not None:
        _draw_confusion_matrix(c, np.array(cms["Decision Tree"]), right_x, top_y - 60, title="Decision Tree")
    c.showPage()

    # ---- ROC Curves ----
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 40, "ROC Curves")
    # drawInlineImage accepts a PIL Image directly
    c.drawInlineImage(roc_img, 50, height/2 - 60, width=520, preserveAspectRatio=True)
    c.showPage()

    # ---- PR Curves ----
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 40, "Precision–Recall Curves")
    c.drawInlineImage(pr_img, 50, height/2 - 60, width=520, preserveAspectRatio=True)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()   # bytes returned


# ---------- Pages ----------
def page_home():
    # Landing Page image
    try:
        landing = Image.open("C:/Users/kojon/PycharmProjects/Text Analytics Projects/Text Anaytics/LANDING IMAGE.png")
        st.image(landing, use_container_width=True)
    except Exception:
        st.info("Add a landing image to show it here.")

    # ---- Centered header + divider ----
    st.markdown(
        "<h2 style='text-align:center;margin:0 0 .25rem 0;'>Navigate</h2>",
        unsafe_allow_html=True
    )
    st.divider()

    # ---- Centered button row ----
    sp_l, c1, c2, c3, sp_r = st.columns([1, 2, 2, 2, 1])

    clicked = None  # collect which button was clicked this run

    with c1:
        if st.button("📊 Dataset", use_container_width=True, type="secondary", key="btn_dataset"):
            clicked = "Dataset"

    with c2:
        if st.button("🛠️ Sarcasm detection", use_container_width=True, type="secondary", key="btn_detect"):
            clicked = "Sarcasm detection"

    with c3:
        if st.button("🧑‍🏫 Prediction – What-If", use_container_width=True, type="secondary", key="btn_predict"):
            clicked = "Prediction"

    # If any button was pressed, update state and rerun (outside of callbacks)
    if clicked:
        st.session_state["selected_page"] = clicked
        st.rerun()



def page1():
    st.subheader("Dataset")
    colA, colB, colC = st.columns([1, 1, 2])
    with colA:
        show_raw = st.checkbox("Show raw headlines", value=True)
    with colB:
        show_clean = st.checkbox("Show cleaned text", value=False)
    with colC:
        n_rows = st.slider("Rows to preview", 10, 200, 20, 10)

    cols_to_show = ["is_sarcastic"]
    if show_raw:
        cols_to_show.insert(0, "headline")
    if show_clean:
        cols_to_show.insert(1 if show_raw else 0, "clean")

    st.dataframe(s_df[cols_to_show].head(n_rows))
    st.bar_chart(s_df["is_sarcastic"].value_counts().sort_index())
    st.caption("Target: 1 = sarcastic, 0 = not sarcastic")

    if st.session_state.get("featurizer") == "tfidf":
        st.info("Using TF-IDF features (1–2 grams, up to 30k features).")
    else:
        st.info(f"Using GloVe averaged embeddings ({GLOVE_DIM}-d) "
                f"{'with' if USE_SCALER else 'without'} standardization.")

def page2():
    st.subheader("Sarcasm detection")

    col1, col2 = st.columns([2, 1])
    with col1:
        model_choice = st.radio(
            "Select model(s) to display",
            ["Logistic Regression", "Decision Tree", "Both"],
            index=2, horizontal=True
        )
    with col2:
        threshold = st.slider("Decision threshold", 0.10, 0.90, st.session_state.get("threshold", 0.50), 0.01, key="slider")

    # Always train both so page3 can always use either
    with st.spinner("Training models..."):
        res_lr = evaluate_model("Logistic Regression", lr_model, X_train_vec, y_train, X_test_vec, y_test, threshold)
        res_dt = evaluate_model("Decision Tree", dt_model, X_train_vec, y_train, X_test_vec, y_test, threshold)

    # Decide which results to SHOW (both are trained)
    to_show = []
    if model_choice in ["Logistic Regression", "Both"]:
        to_show.append(res_lr)
    if model_choice in ["Decision Tree", "Both"]:
        to_show.append(res_dt)

    # Metrics table
    metrics_table = pd.DataFrame([
        {"Model": r["name"], "Accuracy": r["acc"], "Precision": r["prec"],
         "Recall": r["rec"], "F1": r["f1"], "ROC-AUC": r["auc"]}
        for r in to_show
    ]).set_index("Model")
    st.dataframe(metrics_table.style.format("{:.3f}"))

    # ROC Curves
    st.markdown("**ROC Curves**")
    fig = plt.figure()
    for r in to_show:
        fpr, tpr, _ = roc_curve(y_test, r["y_score"])
        plt.plot(fpr, tpr, label=f"{r['name']} (AUC={r['auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    st.pyplot(fig, clear_figure=True)

    # PR Curves
    st.markdown("**Precision–Recall Curves**")
    fig2 = plt.figure()
    for r in to_show:
        pr, rc, _ = precision_recall_curve(y_test, r["y_score"])
        plt.plot(rc, pr, label=r["name"])
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.legend(loc="lower left")
    st.pyplot(fig2, clear_figure=True)

    st.info("Both models were trained. The visuals reflect your selection above.")

    # Cache BOTH models + threshold for page3
    st.session_state["threshold"] = threshold
    st.session_state["lr_model"] = res_lr["model"]
    st.session_state["dt_model"] = res_dt["model"]

    # Record which featurizer these models were trained with
    trained_feat = st.session_state.get("featurizer")
    st.session_state["predict_featurizer"] = trained_feat
    st.session_state["trained_featurizer"] = trained_feat

    # Remember preferred default for page3 dropdown
    st.session_state["active_model_name"] = (
        "Logistic Regression" if model_choice == "Logistic Regression"
        else "Decision Tree" if model_choice == "Decision Tree"
        else "Logistic Regression"
    )

    # ----- Build dataset info & confusion matrices for the report -----
    feat_dim = X_test_vec.shape[1] if hasattr(X_test_vec, "shape") else (len(X_test_vec[0]) if len(X_test_vec) else "N/A")
    counts = s_df["is_sarcastic"].value_counts().sort_index()
    dataset_info = {
        "n_rows": len(s_df),
        "counts_str": f"{int(counts.get(0,0))}/{int(counts.get(1,0))}",
        "featurizer": st.session_state.get("featurizer", "<unknown>"),
        "feat_dim": feat_dim,
        "threshold": threshold,
    }
    cms = {
        "Logistic Regression": res_lr.get("cm"),
        "Decision Tree": res_dt.get("cm"),
    }

    # ----- Generate PDF & CSV buttons -----
    pdf_bytes = generate_pdf_report_enhanced(dataset_info, metrics_table, cms, fig, fig2)
    if pdf_bytes is not None:
        st.download_button(
            label="📄 Download Report (PDF)",
            data=pdf_bytes,                 # BYTES, not BytesIO
            file_name="sarcasm_detection_report.pdf",
            mime="application/pdf"
        )
    st.download_button(
        label="⬇️ Download Metrics (CSV)",
        data=metrics_table.to_csv().encode("utf-8"),
        file_name="metrics_table.csv",
        mime="text/csv"
    )

def page3():
    st.subheader("Prediction – What-If Lab")

    # Pick which trained model to use
    available = ["Logistic Regression", "Decision Tree"]
    default_idx = 0
    if st.session_state.get("active_model_name") in available:
        default_idx = available.index(st.session_state["active_model_name"])
    which_model = st.selectbox("Model for prediction", available, index=default_idx)

    # Live-edit headline
    col_in, col_ctrl = st.columns([3, 2])
    with col_in:
        txt_input = st.text_area(
            "Edit the headline (updates on predict):",
            "local man wins lottery, buys spaceship",
            height=100
        )
    with col_ctrl:
        thr = st.slider("Decision threshold", 0.10, 0.90, st.session_state.get("threshold", 0.50), 0.01)
        st.markdown("**Quick what-ifs**")
        if st.button("Remove punctuation"):
            txt_input = re.sub(r"[^\w\s]", " ", txt_input)
        if st.button("Lowercase"):
            txt_input = txt_input.lower()
        if st.button("Remove numbers"):
            txt_input = re.sub(r"\d+", " ", txt_input)

    # Ensure trained model
    mdl = st.session_state.get("lr_model") if which_model == "Logistic Regression" else st.session_state.get("dt_model")
    if mdl is None:
        st.warning("No trained model found. Go to 'Sarcasm detection' (Page 2) and train first.")
        return

    if st.button("Predict"):
        trained_feat = st.session_state.get("trained_featurizer")
        if trained_feat is None:
            st.warning("No trained featurizer found. Train models on Page 2 first.")
            return

        text_clean = clean_text(txt_input)
        vec = _vectorize_text_for_prediction(text_clean, trained_feat)
        if vec is None:
            return

        proba = _predict_proba(mdl, vec)
        pred = int(proba >= thr)

        st.metric(
            label="Probability (sarcastic)",
            value=f"{proba:.3f}",
            delta="sarcastic" if pred == 1 else "not sarcastic",
            delta_color="inverse" if pred == 0 else "normal"
        )

        # Token-level What-If
        contrib_df = _token_contributions(text_clean, mdl, trained_feat)
        if contrib_df is not None and not contrib_df.empty:
            st.markdown("**Which words matter most?** (Δ = probability drop when the word is removed)")
            st.dataframe(contrib_df.head(15).style.format({"delta_drop": "{:.4f}", "new_proba": "{:.3f}"}))

            st.markdown("**Highlighted headline** (strong push → deeper color)")
            st.markdown(_colorize_text_by_contrib(text_clean, contrib_df), unsafe_allow_html=True)

            removable = contrib_df["token"].tolist()
            to_remove = st.multiselect("Remove these words and recompute", removable[:10])
            if st.button("Apply removals"):
                edited = " ".join([t for t in text_clean.split() if t not in set(to_remove)])
                vec2 = _vectorize_text_for_prediction(edited, trained_feat)
                if vec2 is not None:
                    proba2 = _predict_proba(mdl, vec2)
                    st.write(
                        pd.DataFrame(
                            [{"Version": "Original", "Prob(sarcastic)": proba},
                             {"Version": "Edited",   "Prob(sarcastic)": proba2}]
                        ).style.format({"Prob(sarcastic)": "{:.3f}"})
                    )
        else:
            st.info("No informative tokens found for contribution view (very short or all OOV).")

# ---------- Helpers used by Page 3 ----------
def _vectorize_text_for_prediction(text_clean: str, trained_feat: str):
    if trained_feat == "tfidf":
        tfidf = st.session_state.get("tfidf")
        if tfidf is None:
            st.error("TF-IDF vectorizer missing. Train on Page 2 first.")
            return None
        return tfidf.transform([text_clean])
    else:
        glove = st.session_state.get("glove")
        dim = st.session_state.get("glove_dim")
        scaler = st.session_state.get("glove_scaler")
        if glove is None or dim is None:
            st.error("GloVe resources missing. Train on Page 2 first.")
            return None
        vec_arr = transform_texts_to_glove([text_clean], glove, dim)
        if scaler is not None:
            vec_arr = scaler.transform(vec_arr)
        return vec_arr

def _predict_proba(model, vec) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(vec)[0, 1])
    if hasattr(model, "decision_function"):
        score = float(model.decision_function(vec)[0])
        return 1.0 / (1.0 + np.exp(-score))
    return float(model.predict(vec)[0])

def _token_contributions(text_clean: str, model, trained_feat: str) -> pd.DataFrame | None:
    tokens = text_clean.split()
    if len(tokens) == 0:
        return None
    vec_orig = _vectorize_text_for_prediction(text_clean, trained_feat)
    if vec_orig is None:
        return None
    p_orig = _predict_proba(model, vec_orig)

    rows = []
    for i, tok in enumerate(tokens):
        edited = " ".join([t for j, t in enumerate(tokens) if j != i])
        vec_i = _vectorize_text_for_prediction(edited, trained_feat)
        if vec_i is None:
            continue
        p_new = _predict_proba(model, vec_i)
        rows.append({"token": tok, "new_proba": p_new, "delta_drop": p_orig - p_new})
    if not rows:
        return None
    df = pd.DataFrame(rows).groupby("token", as_index=False).mean()
    df = df.sort_values("delta_drop", ascending=False, ignore_index=True)
    return df

def _colorize_text_by_contrib(text_clean: str, contrib_df: pd.DataFrame) -> str:
    toks = text_clean.split()
    impact = {r["token"]: float(r["delta_drop"]) for _, r in contrib_df.iterrows()}
    max_imp = max(impact.values()) if impact else 0.0

    def color_for(tok):
        val = impact.get(tok, 0.0)
        if max_imp <= 1e-8:
            alpha = 0.0
        else:
            alpha = min(max(val / max_imp, 0.0), 1.0)
        return f'<span style="background-color: rgba(255, 0, 0, {0.15 + 0.35*alpha}); padding:2px; border-radius:4px; margin:1px;">{tok}</span>'

    colored = " ".join(color_for(t) for t in toks)
    return f"<div style='line-height:2.0; font-size:1.05rem'>{colored}</div>"

# ---------- Router ----------
pages = {
    "Home": page_home,
    "Dataset": page1,
    "Sarcasm detection": page2,
    "Prediction": page3,
}

# Current page from session_state
current = st.session_state.get("selected_page", "Home")
page_names = list(pages.keys())
default_idx = page_names.index(current) if current in page_names else 0

def _from_sidebar():
    # No rerun here—selectbox changes already trigger a rerun automatically
    st.session_state["selected_page"] = st.session_state["page_select"]

st.sidebar.selectbox(
    "Select Page",
    page_names,
    index=default_idx,
    key="page_select",
    on_change=_from_sidebar,
)

# Render selected page
pages[st.session_state.get("selected_page", "Home")]()