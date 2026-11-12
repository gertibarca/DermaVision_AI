from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dermavision.config import (
    APP_NAME,
    APP_VERSION,
    ATTENTION_THRESHOLD,
    BENCHMARK,
    CLASS_METRICS,
    CLASSES,
    LESION_INFO,
    MODEL_PATH,
    SAMPLE_LABELS,
    SAMPLES_DIR,
)
from dermavision.database import AnalysisRepository
from dermavision.model import ModelService
from dermavision.quality import assess_image_quality
from dermavision.report import create_pdf_report
from dermavision.ui import hero, inject_styles, probability_chart, result_banner
from dermavision.utils import read_image, sha256

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()


@st.cache_resource(show_spinner=False)
def get_model() -> ModelService:
    return ModelService()


@st.cache_resource(show_spinner=False)
def get_repository() -> AnalysisRepository:
    return AnalysisRepository()


repository = get_repository()

with st.sidebar:
    st.markdown("## 🔬 DermaVision AI")
    st.caption("Analizë e shpjegueshme e lezioneve")
    st.markdown("---")
    page = st.radio(
        "Navigimi",
        ("Analiza", "Historia", "Performanca", "Rreth projektit"),
        label_visibility="collapsed",
    )
    st.markdown("---")
    try:
        with st.spinner("Duke ngarkuar modelin..."):
            model = get_model()
        st.success("Modeli është aktiv")
        st.caption(f"Përpunimi: {model.device_label}")
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        model = None
        st.error("Modeli nuk u ngarkua")
        if not MODEL_PATH.exists():
            st.caption("Ekzekuto download_model.py një herë.")
        else:
            st.caption(str(error))
    st.markdown("---")
    st.caption(f"Versioni {APP_VERSION} · Projekt edukativ")


def render_analysis_page() -> None:
    hero(
        "Analizë inteligjente e lezioneve",
        "ResNet18, test-time augmentation, kontroll i cilësisë, pasiguri e normalizuar dhe Grad-CAM në një rrjedhë të vetme.",
        "Computer vision · Explainable AI",
    )
    st.markdown(
        '<div class="notice"><b>Kufizim i rëndësishëm:</b> Ky model është trajnuar me '
        "imazhe dermatoskopike HAM10000. Fotot nga telefoni mund të kenë saktësi më të ulët. "
        "Rezultati nuk është diagnozë mjekësore.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.subheader("1 · Zgjidh imazhin")
        source = st.radio(
            "Burimi",
            ("Ngarko skedar", "Përdor kamerën", "Shembujt e projektit"),
            horizontal=True,
            label_visibility="collapsed",
        )
        upload = None
        sample_name = None
        if source == "Ngarko skedar":
            upload = st.file_uploader(
                "JPG, PNG ose WEBP · maksimumi 8 MB",
                type=("jpg", "jpeg", "png", "webp"),
            )
        elif source == "Përdor kamerën":
            upload = st.camera_input("Fotografo lezionin")
        else:
            sample_name = st.selectbox(
                "Zgjidh një shembull HAM10000",
                tuple(SAMPLE_LABELS),
                format_func=lambda name: (
                    f"{name.removesuffix('.jpg')} · "
                    f"{LESION_INFO[SAMPLE_LABELS[name]].name_sq}"
                ),
            )

    with right:
        st.subheader("2 · Konteksti opsional")
        age_group = st.selectbox(
            "Grupmosha",
            ("", "0–17 vjeç", "18–29 vjeç", "30–44 vjeç", "45–59 vjeç", "60+ vjeç"),
        )
        sex = st.selectbox("Gjinia", ("", "Femër", "Mashkull", "Tjetër / pa përgjigje"))
        body_site = st.selectbox(
            "Zona e trupit",
            ("", "Kokë / qafë", "Trup", "Krah", "Dorë", "Këmbë", "Shputë", "Tjetër"),
        )
        metadata = {"age_group": age_group, "sex": sex, "body_site": body_site}

    if sample_name:
        filename = sample_name
        raw = (SAMPLES_DIR / sample_name).read_bytes()
        known_label = SAMPLE_LABELS[sample_name]
    elif upload is not None:
        raw = upload.getvalue()
        filename = getattr(upload, "name", "camera.jpg")
        known_label = None
    else:
        st.info("Ngarko një imazh dermatoskopik për të filluar analizën.")
        return
    try:
        image = read_image(raw)
    except ValueError as error:
        st.error(str(error))
        return

    digest = sha256(raw)
    quality = assess_image_quality(image)

    st.markdown("### 3 · Kontrolli para analizës")
    preview, checks = st.columns([1.05, 0.95], gap="large")
    with preview:
        st.image(
            image,
            caption=f"{filename} · {image.width} × {image.height} px",
            use_container_width=True,
        )
    with checks:
        q1, q2 = st.columns(2)
        q1.metric("Cilësia", f"{quality.score}/100")
        q2.metric("Rezolucioni", f"{quality.width}×{quality.height}")
        q3, q4 = st.columns(2)
        q3.metric("Fokusi", f"{quality.sharpness:.0f}")
        q4.metric("Ndriçimi", f"{quality.brightness:.0f}/255")
        if quality.issues:
            for issue in quality.issues:
                st.warning(issue)
        else:
            st.success("Imazhi kaloi të gjitha kontrollet teknike.")
        if known_label:
            st.info(
                f"Etiketa reale e dataset-it: "
                f"{LESION_INFO[known_label].name_sq} ({known_label})"
            )

    override = False
    if not quality.looks_like_skin:
        override = st.checkbox(
            "E kuptoj paralajmërimin dhe dua ta analizoj vetëm për demonstrim."
        )
    consent = st.checkbox(
        "E kuptoj se ky është sistem edukativ dhe rezultati nuk është diagnozë."
    )
    allowed = consent and not quality.blocking and (quality.looks_like_skin or override)
    analyze = st.button(
        "🔬 Analizo me DermaVision AI",
        type="primary",
        use_container_width=True,
        disabled=not allowed or model is None,
    )

    if analyze:
        with st.spinner("Po analizohen 4 orientime dhe po krijohet Grad-CAM..."):
            prediction = model.predict(image)
            heatmap, overlay = model.gradcam(image, prediction.predicted_code)
            record_id = repository.save(
                image_hash=digest,
                filename=filename,
                prediction=prediction,
                quality=quality,
                metadata=metadata,
            )
            st.session_state["analysis"] = {
                "digest": digest,
                "image": image,
                "quality": quality,
                "prediction": prediction,
                "heatmap": heatmap,
                "overlay": overlay,
                "metadata": metadata,
                "record_id": record_id,
            }

    result = st.session_state.get("analysis")
    if not result or result["digest"] != digest:
        return

    prediction = result["prediction"]
    lesion = LESION_INFO[prediction.predicted_code]
    st.markdown("---")
    st.markdown("## Rezultati i analizës")
    result_banner(prediction)
    st.write("")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Kandidati kryesor", lesion.name_sq)
    m2.metric("Besueshmëria", f"{prediction.confidence:.1%}")
    m3.metric("Sinjali i vëmendjes", f"{prediction.attention_score:.1%}")
    m4.metric("Pasiguria", prediction.uncertainty_label)

    tab1, tab2, tab3 = st.tabs(("Shpjegimi vizual", "Probabilitetet", "Interpretimi"))
    with tab1:
        a, b, c = st.columns(3)
        a.image(result["image"], caption="Imazhi origjinal", use_container_width=True)
        b.image(result["heatmap"], caption="Grad-CAM", use_container_width=True)
        c.image(result["overlay"], caption="Mbivendosja", use_container_width=True)
        st.caption(
            "E kuqja tregon zonat që ndikuan më shumë te klasa kryesore. Grad-CAM "
            "shpjegon vëmendjen e modelit, jo kufirin e sëmundjes."
        )
    with tab2:
        st.plotly_chart(probability_chart(prediction), use_container_width=True)
        frame = pd.DataFrame(
            [
                {
                    "Kodi": code,
                    "Klasa": LESION_INFO[code].name_sq,
                    "Probabiliteti": value,
                }
                for code, value in sorted(
                    prediction.probabilities.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]
        )
        st.dataframe(
            frame,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Probabiliteti": st.column_config.ProgressColumn(
                    "Probabiliteti", min_value=0.0, max_value=1.0, format="%.3f"
                )
            },
        )
    with tab3:
        st.markdown(f"### {lesion.name_sq}")
        st.caption(f"{lesion.name_en} · kodi HAM10000: {prediction.predicted_code}")
        st.write(lesion.description)
        st.info(lesion.guidance)
        st.markdown(
            "**Rregulli i sinjalit:** probabilitetet `akiec + bcc + mel` mblidhen dhe "
            f"krahasohen me pragun {ATTENTION_THRESHOLD:.0%}. Pragu favorizon recall-in, "
            "prandaj mund të prodhojë alarme të rreme."
        )

    pdf = create_pdf_report(
        result["image"],
        result["overlay"],
        prediction,
        result["quality"],
        result["metadata"],
    )
    st.download_button(
        "⬇️ Shkarko raportin PDF",
        data=pdf,
        file_name=f"DermaVision_raport_{result['record_id']}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def render_history_page() -> None:
    hero(
        "Historia lokale e analizave",
        "Rezultatet ruhen në SQLite pa ruajtur fotografitë. Kjo mbron privatësinë dhe mundëson statistika.",
        "Data layer · SQLite",
    )
    history = pd.DataFrame(repository.recent())
    if history.empty:
        st.info("Nuk ka ende analiza të ruajtura.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Analiza", len(history))
    c2.metric("Sinjale vëmendjeje", int(history["requires_attention"].sum()))
    c3.metric("Besueshmëria mesatare", f"{history['confidence'].mean():.1%}")
    c4.metric("Cilësia mesatare", f"{history['quality_score'].mean():.0f}/100")

    left, right = st.columns(2, gap="large")
    with left:
        counts = history["predicted_code"].value_counts().reset_index()
        counts.columns = ["Kodi", "Numri"]
        counts["Klasa"] = counts["Kodi"].map(lambda code: LESION_INFO[code].name_sq)
        figure = px.bar(
            counts,
            x="Numri",
            y="Klasa",
            orientation="h",
            title="Shpërndarja e klasave",
            color="Kodi",
            color_discrete_map={code: LESION_INFO[code].color for code in CLASSES},
        )
        figure.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(figure, use_container_width=True)
    with right:
        figure = px.scatter(
            history,
            x="confidence",
            y="attention_score",
            color=history["requires_attention"].map({0: "Pa sinjal", 1: "Vëmendje"}),
            size="quality_score",
            title="Besueshmëria kundrejt sinjalit",
            labels={"confidence": "Besueshmëria", "attention_score": "Sinjali"},
            color_discrete_map={"Pa sinjal": "#10b981", "Vëmendje": "#e11d48"},
        )
        figure.add_hline(y=ATTENTION_THRESHOLD, line_dash="dash", line_color="#e11d48")
        figure.update_layout(paper_bgcolor="rgba(0,0,0,0)", legend_title_text="Statusi")
        st.plotly_chart(figure, use_container_width=True)

    display = history.copy()
    display["Klasa"] = display["predicted_code"].map(
        lambda code: LESION_INFO[code].name_sq
    )
    display["Statusi"] = display["requires_attention"].map(
        {1: "Kërkon vëmendje", 0: "Pa sinjal të fortë"}
    )
    display = display.rename(
        columns={
            "created_at": "Data (UTC)",
            "filename": "Skedari",
            "confidence": "Besueshmëria",
            "attention_score": "Sinjali",
            "quality_score": "Cilësia",
        }
    )
    visible = [
        "id",
        "Data (UTC)",
        "Skedari",
        "Klasa",
        "Besueshmëria",
        "Sinjali",
        "Cilësia",
        "Statusi",
    ]
    st.dataframe(display[visible], hide_index=True, use_container_width=True)
    st.download_button(
        "Shkarko historinë CSV",
        display.to_csv(index=False).encode("utf-8-sig"),
        "dermavision_historia.csv",
        "text/csv",
    )
    with st.expander("Zona e rrezikshme"):
        confirm = st.checkbox("Konfirmoj fshirjen e të gjithë historikut")
        if st.button("Fshi historikun", disabled=not confirm):
            repository.clear()
            st.session_state.pop("analysis", None)
            st.rerun()


def render_performance_page() -> None:
    hero(
        "Performanca dhe transparenca",
        "Metrika reale nga një test set prej 1,508 imazhesh, i ndarë sipas lesion_id për të shmangur rrjedhjen e të njëjtit lezion.",
        "Model card · Responsible AI",
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Imazhe testimi", f"{BENCHMARK['test_images']:,}")
    c2.metric("Accuracy 7-klasësh", f"{BENCHMARK['accuracy']:.1%}")
    c3.metric("Recall i sinjalit", f"{BENCHMARK['attention_recall']:.1%}")
    c4.metric("Precision i sinjalit", f"{BENCHMARK['attention_precision']:.1%}")
    st.warning(
        "Accuracy nuk mjafton: shumica e dataset-it janë nevus. Prandaj sistemi shfaq "
        "recall, precision, pasigurinë dhe të gjitha probabilitetet."
    )

    rows = [
        {
            "Kodi": code,
            "Klasa": LESION_INFO[code].name_sq,
            "Recall": metric["recall"],
            "Precision": metric["precision"],
            "Support": metric["support"],
        }
        for code, metric in CLASS_METRICS.items()
    ]
    metrics = pd.DataFrame(rows)
    melted = metrics.melt(
        id_vars=("Kodi", "Klasa"),
        value_vars=("Recall", "Precision"),
        var_name="Metrika",
        value_name="Vlera",
    )
    figure = px.bar(
        melted,
        x="Klasa",
        y="Vlera",
        color="Metrika",
        barmode="group",
        color_discrete_map={"Recall": "#0f766e", "Precision": "#38bdf8"},
    )
    figure.update_yaxes(tickformat=".0%", range=[0, 1.05])
    figure.update_layout(paper_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-25)
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(
        metrics,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Recall": st.column_config.ProgressColumn(
                "Recall", min_value=0, max_value=1, format="%.3f"
            ),
            "Precision": st.column_config.ProgressColumn(
                "Precision", min_value=0, max_value=1, format="%.3f"
            ),
        },
    )

    st.markdown("### Vendime teknike")
    d1, d2, d3 = st.columns(3)
    d1.info(
        "**Transfer learning**\n\nResNet18 i para-trajnuar në ImageNet dhe i përshtatur në HAM10000."
    )
    d2.info("**TTA ×4**\n\nKatër rrotullime mesatarizohen për inferencë më stabile.")
    d3.info("**Grad-CAM**\n\nTregon zonat që ndikuan te klasa e parashikuar.")


def render_about_page() -> None:
    hero(
        "Rreth DermaVision AI",
        "Projekt final në Python që kombinon computer vision, deep learning, explainable AI, validim hyrjeje, databazë dhe raportim.",
        "Python · PyTorch · Streamlit",
    )
    st.markdown("### Arkitektura")
    st.code(
        """
Imazhi JPG/PNG
      │
      ▼
Validim sigurie + kontroll cilësie
      │
      ▼
Resize 224×224 + ImageNet normalization
      │
      ▼
ResNet18 × 4 orientime (Test-Time Augmentation)
      │
      ├──► 7 probabilitete + pasiguri
      ├──► Sinjali i kombinuar: akiec + bcc + mel
      ├──► Grad-CAM explainability
      ├──► SQLite history
      └──► Raport PDF / eksport CSV
        """,
        language="text",
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Dataset-i")
        st.write(
            "HAM10000 përmban 10,015 imazhe dermatoskopike në shtatë kategori. "
            "Modeli përdoret vetëm për qëllime edukative dhe jo-komerciale."
        )
        st.markdown("### Privatësia")
        st.write(
            "Fotografia përpunohet në memorien e aplikacionit dhe nuk ruhet në databazë. "
            "Ruhet vetëm hash-i, rezultati, cilësia dhe metadata opsionale."
        )
    with right:
        st.markdown("### Kufizimet")
        st.write(
            "Modeli është trajnuar me dermatoskop, jo me fotografi telefoni. Klasat e "
            "rralla kanë më pak shembuj dhe performanca mund të ndryshojë sipas pajisjes, "
            "ndriçimit dhe ngjyrës së lëkurës."
        )
        st.markdown("### Etika")
        st.write(
            "Aplikacioni nuk thotë ‘ke kancer’. Ai komunikon pasigurinë dhe e drejton "
            "përdoruesin te profesionisti."
        )


if page == "Analiza":
    render_analysis_page()
elif page == "Historia":
    render_history_page()
elif page == "Performanca":
    render_performance_page()
else:
    render_about_page()

st.markdown("---")
st.caption(
    "DermaVision AI · Projekt edukativ · Jo pajisje mjekësore · Çdo lezion i ri ose "
    "që ndryshon duhet vlerësuar nga dermatologu."
)
