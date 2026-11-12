from __future__ import annotations

import io
from datetime import datetime
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as ReportImage,
)
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .config import LESION_INFO
from .model import Prediction
from .quality import QualityResult


def _pil_buffer(image: Image.Image) -> io.BytesIO:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    return buffer


def create_pdf_report(
    image: Image.Image,
    overlay: Image.Image,
    prediction: Prediction,
    quality: QualityResult,
    metadata: dict[str, str] | None = None,
) -> bytes:
    metadata = metadata or {}
    output = io.BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Raport edukativ - DermaVision AI",
        author="DermaVision AI",
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleDV",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=colors.HexColor("#102A43"),
        alignment=TA_CENTER,
        spaceAfter=7,
    )
    subtitle = ParagraphStyle(
        "SubtitleDV",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#627D98"),
        alignment=TA_CENTER,
        leading=13,
    )
    heading = ParagraphStyle(
        "HeadingDV",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=colors.HexColor("#0F766E"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "BodyDV",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#243B53"),
        alignment=TA_LEFT,
    )
    warning = ParagraphStyle(
        "WarningDV",
        parent=body,
        backColor=colors.HexColor("#FFF7ED"),
        borderColor=colors.HexColor("#FDBA74"),
        borderWidth=0.8,
        borderPadding=9,
        textColor=colors.HexColor("#9A3412"),
        spaceBefore=8,
        spaceAfter=8,
    )

    info = LESION_INFO[prediction.predicted_code]
    status = (
        "Sinjal që kërkon vlerësim profesional"
        if prediction.requires_attention
        else "Nuk u zbulua sinjal i fortë nga modeli"
    )

    story = [
        Paragraph("DermaVision AI", title),
        Paragraph(
            f"Raport edukativ i analizës · {datetime.now().strftime('%d.%m.%Y, %H:%M')}",
            subtitle,
        ),
        Spacer(1, 0.35 * cm),
        Paragraph(
            "KUJDES: Ky sistem është projekt edukativ dhe nuk është pajisje mjekësore. "
            "Rezultati nuk është diagnozë dhe nuk zëvendëson dermatologun ose biopsinë.",
            warning,
        ),
        Paragraph("Përmbledhja e rezultatit", heading),
    ]

    summary_data = [
        ["Statusi", status],
        ["Klasa e sugjeruar", f"{info.name_sq} ({prediction.predicted_code})"],
        ["Besueshmëria e klasës", f"{prediction.confidence:.1%}"],
        ["Sinjali i kombinuar i vëmendjes", f"{prediction.attention_score:.1%}"],
        [
            "Pasiguria e modelit",
            f"{prediction.uncertainty_label} ({prediction.uncertainty:.1%})",
        ],
        ["Cilësia teknike e imazhit", f"{quality.score}/100"],
    ]
    summary = Table(summary_data, colWidths=[6.0 * cm, 10.0 * cm])
    summary.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E6FFFA")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#243B53")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#BCCCDC")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(summary)

    if any(metadata.values()):
        story.extend(
            [
                Paragraph("Të dhëna opsionale", heading),
                Paragraph(
                    " · ".join(
                        escape(value)
                        for value in (
                            metadata.get("age_group", ""),
                            metadata.get("sex", ""),
                            metadata.get("body_site", ""),
                        )
                        if value
                    ),
                    body,
                ),
            ]
        )

    story.extend(
        [
            Paragraph("Imazhi dhe shpjegimi Grad-CAM", heading),
            KeepTogether(
                Table(
                    [
                        [
                            ReportImage(
                                _pil_buffer(image), width=7.5 * cm, height=7.5 * cm
                            ),
                            ReportImage(
                                _pil_buffer(overlay), width=7.5 * cm, height=7.5 * cm
                            ),
                        ],
                        [
                            Paragraph("Imazhi i analizuar", subtitle),
                            Paragraph("Zonat që ndikuan te modeli", subtitle),
                        ],
                    ],
                    colWidths=[8.0 * cm, 8.0 * cm],
                    hAlign="CENTER",
                )
            ),
            Paragraph("Interpretimi", heading),
            Paragraph(
                f"<b>{escape(info.name_sq)}</b> — {escape(info.description)}", body
            ),
            Spacer(1, 0.15 * cm),
            Paragraph(escape(info.guidance), body),
            Paragraph("Probabilitetet e shtatë klasave", heading),
        ]
    )

    probability_rows = [["Klasa", "Probabiliteti"]]
    probability_rows.extend(
        [LESION_INFO[code].name_sq, f"{probability:.2%}"]
        for code, probability in sorted(
            prediction.probabilities.items(), key=lambda item: item[1], reverse=True
        )
    )
    probabilities = Table(probability_rows, colWidths=[12.5 * cm, 3.5 * cm])
    probabilities.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F0FDFA")],
                ),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(probabilities)
    story.extend(
        [
            Spacer(1, 0.3 * cm),
            Paragraph(
                "Nëse lezioni është i ri, po ndryshon, kruhet, dhemb ose gjakoset, kontakto një dermatolog "
                "pavarësisht rezultatit të këtij modeli.",
                warning,
            ),
        ]
    )

    document.build(story)
    return output.getvalue()
