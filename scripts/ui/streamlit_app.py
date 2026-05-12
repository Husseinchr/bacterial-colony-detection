from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui.inference import (
    CLASSICAL_COUNT_PRESETS,
    CLASSICAL_SPECIES_MODEL_PRESETS,
    bgr_to_rgb,
    classify_with_classical_species_model,
    config_rows,
    count_with_classical_model,
    load_image_input,
    mask_to_rgb,
)


st.set_page_config(page_title="Bacterial Colony Model Tester", page_icon="BC", layout="wide")


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        #MainMenu, footer, header[data-testid="stHeader"], div[data-testid="stToolbar"] {
            visibility: hidden;
            height: 0;
        }
        .stApp {
            background: #eef2ef;
            color: #16211c;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 1.6rem;
            padding-bottom: 2.2rem;
        }
        section[data-testid="stSidebar"] {
            background: #16211c;
            border-right: 1px solid #2b3c33;
        }
        section[data-testid="stSidebar"] * {
            color: #edf5ef;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea {
            background: #223128;
            border-color: #40584b;
            color: #f7fbf8;
        }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {
            color: #dce8e0;
        }
        div[data-testid="stFileUploader"] section {
            background: #223128;
            border: 1px dashed #5d7468;
            border-radius: 8px;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid #d9e3dd;
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 8px 22px rgba(23, 33, 28, 0.055);
        }
        h1, h2, h3 {
            color: #16211c;
            letter-spacing: 0;
        }
        h1 {
            font-size: 1.9rem;
            line-height: 1.12;
            margin-bottom: 0.15rem;
        }
        h2 {
            font-size: 1.12rem;
        }
        h3 {
            font-size: 1rem;
        }
        .app-shell {
            background: #ffffff;
            border: 1px solid #d9e3dd;
            border-radius: 8px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 10px 26px rgba(23, 33, 28, 0.06);
            margin-bottom: 1rem;
        }
        .app-title {
            color: #16211c;
            font-size: 1.9rem;
            line-height: 1.12;
            font-weight: 800;
            margin-bottom: 0.15rem;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.85rem;
        }
        .phase-line {
            color: #5d6c64;
            font-size: 0.94rem;
            margin-bottom: 1rem;
        }
        .status-card {
            background: #f4f7f4;
            border: 1px solid #dce6df;
            border-left: 4px solid #2f7d5b;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            min-height: 5.15rem;
        }
        .status-label {
            color: #64736b;
            font-size: 0.78rem;
            text-transform: uppercase;
            font-weight: 700;
        }
        .status-value {
            color: #16211c;
            font-size: 1.04rem;
            font-weight: 700;
            margin-top: 0.22rem;
        }
        .status-detail {
            color: #65746c;
            font-size: 0.86rem;
            margin-top: 0.14rem;
        }
        .result-title {
            color: #16211c;
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .result-muted {
            color: #697b72;
            font-size: 0.9rem;
        }
        .empty-state {
            background: #f5f7f6;
            border: 1px dashed #c4d1ca;
            border-radius: 8px;
            color: #5e6f66;
            padding: 1.1rem;
            min-height: 3.5rem;
        }
        div.stButton > button {
            border-radius: 6px;
            font-weight: 700;
            background: #2f7d5b;
            border-color: #2f7d5b;
            color: #ffffff;
        }
        div.stButton > button:hover {
            background: #25694c;
            border-color: #25694c;
            color: #ffffff;
        }
        div[data-testid="stMetric"] {
            background: #f6f8f7;
            border: 1px solid #dfe7e2;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
        }
        div[data-testid="stMetric"] label {
            color: #5f7068;
        }
        div[data-baseweb="select"] > div,
        input,
        textarea {
            border-radius: 7px;
        }
        @media (max-width: 900px) {
            .status-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 560px) {
            .status-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_styles()
    task, model_family, uploaded_file, image_path = render_sidebar()
    render_header(task, model_family)
    image, source_name = resolve_input(uploaded_file, image_path)

    if model_family != "Classical":
        render_unavailable_family(task, model_family, image, source_name)
        return

    if task == "Counting":
        render_classical_counting(image, source_name)
    else:
        render_classical_classification(image, source_name)


def render_sidebar():
    with st.sidebar:
        st.header("Run Setup")
        task = st.radio("Task", options=["Counting", "Classification"], horizontal=False)
        model_family = st.selectbox("Model family", options=["Classical", "YOLO", "U-Net"], index=0)
        st.divider()
        image_source = st.radio("Image source", options=["Upload", "Path"], horizontal=True)
        uploaded_file = None
        image_path = ""
        if image_source == "Upload":
            uploaded_file = st.file_uploader("Plate image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
        else:
            image_path = st.text_input("Image path", value="")
    return task, model_family, uploaded_file, image_path


def render_header(task: str, model_family: str) -> None:
    statuses = (
        ("Active task", task, model_family),
        ("Classical count", "Frozen", "config 23 baseline"),
        ("Classical species", "Selected", "val F1 0.5708"),
        ("Next phases", "Pending", "YOLO and U-Net"),
    )
    cards = "\n".join(
        f"""
        <div class="status-card">
            <div class="status-label">{label}</div>
            <div class="status-value">{value}</div>
            <div class="status-detail">{detail}</div>
        </div>
        """
        for label, value, detail in statuses
    )
    st.markdown(
        f"""
        <div class="app-shell">
            <div class="app-title">Bacterial Colony Model Tester</div>
            <div class="phase-line">Single-image workbench for the AGAR-primary classical baselines.</div>
            <div class="status-grid">{cards}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def resolve_input(uploaded_file, image_path: str):
    try:
        uploaded_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
        return load_image_input(image_path=image_path, uploaded_bytes=uploaded_bytes)
    except ValueError:
        return None, ""
    except Exception as exc:
        st.error(str(exc))
        return None, ""


def render_unavailable_family(task: str, model_family: str, image, source_name: str) -> None:
    left, right = st.columns([0.9, 1.1])
    with left:
        with st.container(border=True):
            st.subheader(f"{model_family} {task}")
            st.warning("Backend pending for this phase.")
            st.metric("Implementation status", "Not connected")
            st.metric("Current available family", "Classical")
    with right:
        render_input_panel(image, source_name)


def render_classical_counting(image, source_name: str) -> None:
    control_col, result_col = st.columns([0.9, 1.35])
    with control_col:
        with st.container(border=True):
            st.subheader("Classical Counting")
            preset_name = st.selectbox("Counting preset", options=list(CLASSICAL_COUNT_PRESETS.keys()) + ["Config JSON path"])
            config_json_path = st.text_input("Config JSON path", value="") if preset_name == "Config JSON path" else ""
            run = st.button("Run counting", type="primary", use_container_width=True)
            st.caption("Validation baseline is frozen at config 23 unless a saved config JSON is selected.")
    with result_col:
        render_input_panel(image, source_name)
        with st.container(border=True):
            st.markdown('<div class="result-title">Counting Result</div>', unsafe_allow_html=True)
            if not run:
                st.markdown('<div class="result-muted">No counting run in this session.</div>', unsafe_allow_html=True)
                return
            if image is None:
                st.error("Provide an image before running counting.")
                return
            try:
                result = count_with_classical_model(image=image, preset_name=preset_name, config_json_path=config_json_path)
            except Exception as exc:
                st.error(str(exc))
                return
            metric_a, metric_b = st.columns(2)
            metric_a.metric("Predicted count", result.predicted_count)
            metric_b.metric("Image source", source_name)
            view_a, view_b = st.columns(2)
            view_a.image(bgr_to_rgb(result.overlay_bgr), caption="Overlay", use_container_width=True)
            view_b.image(mask_to_rgb(result.mask), caption="Binary mask", use_container_width=True)
            render_config_table(result.config)


def render_classical_classification(image, source_name: str) -> None:
    control_col, result_col = st.columns([0.9, 1.35])
    with control_col:
        with st.container(border=True):
            st.subheader("Classical Classification")
            preset_names = list(CLASSICAL_SPECIES_MODEL_PRESETS.keys()) + ["Manual path"]
            preset_name = st.selectbox("Species model", options=preset_names)
            default_path = CLASSICAL_SPECIES_MODEL_PRESETS.get(preset_name, "")
            model_json_path = st.text_input("Model JSON path", value=default_path)
            run = st.button("Run classification", type="primary", use_container_width=True)
            st.caption("Use the best validation-selected classical species model.")
    with result_col:
        render_input_panel(image, source_name)
        with st.container(border=True):
            st.markdown('<div class="result-title">Classification Result</div>', unsafe_allow_html=True)
            if not run:
                st.markdown('<div class="result-muted">No classification run in this session.</div>', unsafe_allow_html=True)
                return
            if image is None:
                st.error("Provide an image before running classification.")
                return
            try:
                result = classify_with_classical_species_model(image=image, model_json_path=model_json_path)
            except Exception as exc:
                st.error(str(exc))
                return
            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Predicted class", result.predicted_class)
            metric_b.metric("Classifier", result.classifier_type)
            metric_c.metric("Distance", f"{result.distance:.3f}")
            st.caption(f"Image source: {source_name}")
            render_config_table(result.config)


def render_input_panel(image, source_name: str) -> None:
    with st.container(border=True):
        st.subheader("Input Image")
        if image is None:
            st.markdown('<div class="empty-state">Upload a plate image or provide a local image path.</div>', unsafe_allow_html=True)
            return
        st.image(bgr_to_rgb(image), caption=source_name, use_container_width=True)


def render_config_table(config: dict) -> None:
    st.dataframe(pd.DataFrame(config_rows(config)), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
