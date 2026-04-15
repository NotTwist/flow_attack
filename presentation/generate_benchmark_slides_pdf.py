#!/usr/bin/env python3
"""Generate PDF slides for the video colorization benchmark paper."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# 16:9 landscape (points)
W, H = 792, 445  # ~11" x 6.2"


def register_fonts():
    try:
        pdfmetrics.registerFont(TTFont("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
        return "DejaVu", "DejaVu-Bold"
    except Exception:
        return "Helvetica", "Helvetica-Bold"


def make_slide_canvas(font_bold_name: str):
    def slide_canvas(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#1a365d"))
        canvas.setLineWidth(3)
        canvas.line(36, H - 48, W - 36, H - 48)
        canvas.setFillColor(colors.HexColor("#2c5282"))
        canvas.setFont(font_bold_name, 9)
        canvas.drawRightString(W - 36, 24, "Muravlev et al. — Video Colorization Benchmark")
        canvas.restoreState()

    return slide_canvas


def build_story(font_name, font_bold):
    styles = getSampleStyleSheet()

    title = ParagraphStyle(
        "T",
        parent=styles["Normal"],
        fontName=font_bold,
        fontSize=26,
        leading=30,
        textColor=colors.HexColor("#1a365d"),
        alignment=TA_CENTER,
        spaceAfter=14,
    )
    subtitle = ParagraphStyle(
        "ST",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#4a5568"),
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    h1 = ParagraphStyle(
        "H1",
        parent=styles["Normal"],
        fontName=font_bold,
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#1a365d"),
        alignment=TA_LEFT,
        spaceAfter=16,
    )
    body = ParagraphStyle(
        "B",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#2d3748"),
        alignment=TA_LEFT,
        spaceAfter=10,
    )
    bullet = ParagraphStyle(
        "BL",
        parent=body,
        leftIndent=18,
        bulletIndent=8,
        spaceAfter=8,
    )
    small = ParagraphStyle(
        "SM",
        parent=body,
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#718096"),
    )

    story = []

    # --- Slide 1: Title ---
    story.append(Spacer(1, 1.1 * inch))
    story.append(
        Paragraph(
            "A Comprehensive Benchmark for Evaluating<br/>"
            "Video Colorization and Color Propagation Methods",
            title,
        )
    )
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            "Sergey Muravlev, Sergey Lavrushkin, David Chikovani, Dmitriy Vatolin<br/>"
            "Lomonosov Moscow State University · MSU Institute for Artificial Intelligence",
            subtitle,
        )
    )
    story.append(PageBreak())

    # --- Slide 2: Talk outline ---
    story.append(Spacer(1, 0.35 * inch))
    story.append(Paragraph("Outline", h1))
    for text in [
        "What colorization is and why it matters",
        "Image vs. video colorization (at a glance)",
        "Gaps in evaluation and our benchmark",
        "Dataset, methods, and subjective study",
        "VCQI: a metric aligned with human judgment",
        "Key results and takeaways",
    ]:
        story.append(Paragraph(f"• {text}", bullet))
    story.append(PageBreak())

    # --- Slide 3: What is colorization ---
    story.append(Spacer(1, 0.35 * inch))
    story.append(Paragraph("What is colorization?", h1))
    story.append(
        Paragraph(
            "<b>Definition.</b> Turning grayscale <b>images or video</b> into plausible, "
            "visually realistic color. <i>Color propagation</i> spreads color from reference "
            "frames through a sequence.",
            body,
        )
    )
    story.append(
        Paragraph(
            "<b>Why it is useful.</b> Restoring historical monochrome footage; creative "
            "recoloring in production; reducing reliance on painstaking manual painting.",
            body,
        )
    )
    story.append(
        Paragraph(
            "<b>Why video is harder (preview).</b> A time dimension: colors must stay "
            "<b>stable over time</b>—avoiding flicker, hue drift, and inconsistent objects—"
            "constraints a single frame does not impose. (The next slide contrasts image vs. video.)",
            body,
        )
    )
    story.append(PageBreak())

    # --- Slide 4: Image vs video ---
    story.append(Spacer(1, 0.35 * inch))
    story.append(Paragraph("Image colorization vs. video colorization", h1))
    story.append(
        Paragraph(
            "<b>Image.</b> One spatial decision per output: plausible chroma given luminance "
            "and context; no requirement to match a previous frame.",
            body,
        )
    )
    story.append(
        Paragraph(
            "<b>Video.</b> The same ambiguity exists <i>for every frame</i>, plus "
            "<b>temporal consistency</b>: neighboring frames should agree on object colors, "
            "motion, and lighting—otherwise viewers see distracting temporal artifacts.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Deep learning has made automatic <b>image</b> colorization increasingly practical; "
            "<b>video</b> remains especially difficult because spatial plausibility alone is not enough.",
            body,
        )
    )
    story.append(PageBreak())

    # --- Slide 5: Motivation / gap ---
    story.append(Spacer(1, 0.35 * inch))
    story.append(Paragraph("Why a new benchmark?", h1))
    story.append(
        Paragraph(
            "The field lacked a <b>standardized evaluation framework</b> under diverse, "
            "realistic conditions. Prior work often reused datasets built for other tasks.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Classic objective scores are not validated for colorization and miss perceptual "
            "issues: unnatural bleeding, washout, <b>temporal flicker</b>.",
            body,
        )
    )
    story.append(
        Paragraph(
            "We present the <b>first comprehensive benchmark</b> for video colorization: "
            "curated data, many methods, and a metric trained on human ratings.",
            body,
        )
    )
    story.append(PageBreak())

    # --- Slide 6: Contributions ---
    story.append(Spacer(1, 0.35 * inch))
    story.append(Paragraph("Contributions", h1))
    for text in [
        "<b>New datasets</b> — 153 diverse real-world scenes with subjective quality scores; "
        "clips trimmed so objects do not enter/exit between anchor frames (fair for propagation).",
        "<b>Rigorous comparison</b> — 13 exemplar-based and fully automatic methods.",
        "<b>VCQI</b> — Video Colorization Quality Index: learned from human judgments; "
        "better predicts perceived quality than prior objective metrics alone.",
    ]:
        story.append(Paragraph(f"• {text}", bullet))
    story.append(PageBreak())

    # --- Slide 7: Evaluation setup ---
    story.append(Spacer(1, 0.35 * inch))
    story.append(Paragraph("How we evaluate", h1))
    story.append(
        Paragraph(
            "<b>Objective metrics</b> — PSNR, SSIM, LPIPS, Color, Warp Error, CDC, Inception "
            "distance, and others (chrominance-focused where appropriate).",
            body,
        )
    )
    story.append(
        Paragraph(
            "<b>Large subjective study</b> — crowdsourced pairwise comparisons (Subjective.us); "
            "participants judge saturation, bleeding, flicker; Bradley–Terry scores for ranking.",
            body,
        )
    )
    story.append(
        Paragraph(
            "<b>Throughput</b> — subjective quality vs. FPS to show quality–speed trade-offs.",
            small,
        )
    )
    story.append(PageBreak())

    # --- Slide 8: VCQI ---
    story.append(Spacer(1, 0.35 * inch))
    story.append(Paragraph("VCQI in brief", h1))
    story.append(
        Paragraph(
            "Ridge regression on polynomial-expanded features from selected objectives "
            "(e.g. PSNR, SSIM, Color, block-wise ID). Chosen to maximize correlation with "
            "human ratings.",
            body,
        )
    )
    story.append(Spacer(1, 0.1 * inch))
    data = [
        ["Metric", "Pearson r (vs. subjective)", "Spearman r"],
        ["VCQI (ours)", "0.916", "0.886"],
        ["CDC", "0.865", "0.878"],
        ["PSNR", "0.844", "0.825"],
        ["LPIPS", "0.807", "0.834"],
        ["Warp Error", "0.204", "0.187"],
    ]
    t = Table(data, colWidths=[2.4 * inch, 1.6 * inch, 1.4 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5282")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), font_bold),
                ("FONTSIZE", (0, 0), (-1, -1), 11),
                ("FONTNAME", (0, 1), (-1, -1), font_name),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            "VCQI aligns best with human judgment among compared metrics; some "
            "specialized temporal metrics correlate weakly with subjective scores.",
            small,
        )
    )
    story.append(PageBreak())

    # --- Slide 9: Results summary ---
    story.append(Spacer(1, 0.35 * inch))
    story.append(Paragraph("Main empirical findings", h1))
    for text in [
        "<b>Exemplar-based methods</b> (with anchor frames) generally beat fully automatic ones—"
        "anchor color strongly helps.",
        "<b>ColorMNet</b> and <b>LVVCP</b> rank among the best; several automatic or weak baselines "
        "score below SimpleFilt and are poor choices for robust video colorization.",
        "<b>ColorMNet, DEBVC, DeepRemaster</b> sit on a favorable quality–FPS Pareto front.",
        "<b>BiSTNet</b> (two anchors) did not beat the best single-anchor methods—room to improve "
        "multi-anchor strategies.",
        "<b>Spatial crops</b> (PLCC 0.99 with full-res rankings) enable efficient benchmarking.",
    ]:
        story.append(Paragraph(f"• {text}", bullet))
    story.append(PageBreak())

    # --- Slide 10: Conclusion ---
    story.append(Spacer(1, 0.35 * inch))
    story.append(Paragraph("Conclusion", h1))
    story.append(
        Paragraph(
            "First unified benchmark for video colorization with purpose-built data, "
            "broad method coverage, and VCQI tied to human perception.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Released resources aim to improve <b>reproducibility</b> and drive algorithms that "
            "are both <b>temporally consistent</b> and <b>visually compelling</b>.",
            body,
        )
    )
    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph("Thank you — questions?", ParagraphStyle(
        "TY",
        parent=h1,
        alignment=TA_CENTER,
        fontSize=20,
    )))

    return story


def main():
    font_name, font_bold = register_fonts()
    out_dir = Path(__file__).resolve().parent
    out_pdf = out_dir / "Video_Colorization_Benchmark_Presentation.pdf"

    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=(W, H),
        leftMargin=48,
        rightMargin=48,
        topMargin=52,
        bottomMargin=40,
    )
    story = build_story(font_name, font_bold)
    footer = make_slide_canvas(font_bold)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
