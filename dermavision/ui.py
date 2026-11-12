from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from .config import LESION_INFO
from .model import Prediction


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background: linear-gradient(135deg, #f8fbff 0%, #f5fffd 50%, #f8fafc 100%); }
        [data-testid="stSidebar"] { background: #0b1f33; }
        [data-testid="stSidebar"] * { color: #e7f6ff; }
        [data-testid="stSidebar"] .stRadio label { padding: .35rem 0; }
        .block-container { max-width: 1240px; padding-top: 1.6rem; padding-bottom: 3rem; }
        .hero {
            background: linear-gradient(120deg, #0b1f33, #0f766e);
            padding: 2rem 2.2rem; border-radius: 24px; color: white;
            box-shadow: 0 18px 45px rgba(15, 118, 110, .18); margin-bottom: 1.3rem;
        }
        .hero h1 { color: white; font-size: 2.2rem; margin: 0 0 .35rem 0; }
        .hero p { color: #cdeee9; margin: 0; max-width: 800px; }
        .eyebrow { letter-spacing: .12em; text-transform: uppercase; font-weight: 700; font-size: .72rem; color: #67e8f9; }
        .status-danger { background:#fff1f2; border:1px solid #fecdd3; border-left:6px solid #e11d48; padding:1.1rem 1.2rem; border-radius:14px; }
        .status-safe { background:#ecfdf5; border:1px solid #a7f3d0; border-left:6px solid #059669; padding:1.1rem 1.2rem; border-radius:14px; }
        .status-unknown { background:#fffbeb; border:1px solid #fde68a; border-left:6px solid #d97706; padding:1.1rem 1.2rem; border-radius:14px; }
        .notice { background:#fff7ed; border:1px solid #fed7aa; color:#9a3412; padding:1rem 1.1rem; border-radius:14px; font-size:.92rem; }
        div[data-testid="stMetric"] { background:white; border:1px solid #e2e8f0; padding:1rem; border-radius:16px; box-shadow:0 5px 18px rgba(15,23,42,.04); }
        .stButton > button, .stDownloadButton > button { border-radius:12px; font-weight:700; min-height:2.8rem; }
        footer { visibility:hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, text: str, eyebrow: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def probability_chart(prediction: Prediction) -> go.Figure:
    ordered = sorted(prediction.probabilities.items(), key=lambda item: item[1])
    labels = [LESION_INFO[code].name_sq for code, _ in ordered]
    values = [value * 100 for _, value in ordered]
    bar_colors = [LESION_INFO[code].color for code, _ in ordered]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{value:.1f}%" for value in values],
            textposition="outside",
            hovertemplate="%{y}<br>%{x:.2f}%<extra></extra>",
        )
    )
    figure.update_layout(
        height=390,
        margin={"l": 10, "r": 50, "t": 10, "b": 25},
        xaxis={"title": "Probabiliteti (%)", "range": [0, max(100, max(values) + 8)]},
        yaxis={"title": ""},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return figure


def result_banner(prediction: Prediction) -> None:
    if not prediction.is_confident:
        css_class = "status-unknown"
        title = "Modeli nuk ka siguri të mjaftueshme për ta emërtuar lezionin."
        detail = "Kandidati kryesor shfaqet vetëm për transparencë."
    elif prediction.requires_attention:
        css_class = "status-danger"
        title = "U zbulua sinjal që kërkon vlerësim profesional."
        detail = (
            "Ky rezultat nuk konfirmon kancer; kontakto dermatologun për vlerësim real."
        )
    else:
        css_class = "status-safe"
        title = "Modeli nuk zbuloi sinjal të fortë në klasat që kërkojnë vëmendje."
        detail = "Ky rezultat nuk e përjashton sëmundjen dhe nuk zëvendëson kontrollin."

    st.markdown(
        f'<div class="{css_class}"><b>{title}</b><br>{detail}</div>',
        unsafe_allow_html=True,
    )
