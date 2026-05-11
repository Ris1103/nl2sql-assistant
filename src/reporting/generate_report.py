"""Generate a PDF report pulling real metrics from MLflow after training completes.

Usage:
  python -m src.reporting.generate_report --output nl2sql_project_report.pdf
"""
import argparse
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
)


MLFLOW_URI = (Path(__file__).parents[2] / "mlruns").as_uri()
DARK_BLUE = colors.HexColor("#1a237e")
MID_BLUE = colors.HexColor("#1565c0")
ACCENT = colors.HexColor("#e3f2fd")


def get_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=24, textColor=DARK_BLUE, spaceAfter=6),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=14, textColor=DARK_BLUE, spaceBefore=12, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=11, textColor=MID_BLUE, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["Normal"], fontSize=10, spaceAfter=4),
        "code": ParagraphStyle("code", parent=base["Code"], fontSize=8, backColor=colors.HexColor("#f5f5f5")),
    }


def table_style(header_bg=DARK_BLUE):
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ACCENT]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])


def fetch_mlflow_metrics() -> dict:
    client = MlflowClient(tracking_uri=MLFLOW_URI)

    def get_best_run(experiment_name: str) -> dict:
        try:
            exp = client.get_experiment_by_name(experiment_name)
            if not exp:
                return {}
            runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
            if not runs:
                return {}
            return runs[0].data.metrics
        except Exception:
            return {}

    baseline = get_best_run("nl2sql-baseline")
    finetuned = get_best_run("nl2sql-finetuning")
    return {"baseline": baseline, "finetuned": finetuned}


def generate_loss_curve() -> io.BytesIO | None:
    """Pull training loss from MLflow and return a PNG buffer, or None if unavailable."""
    client = MlflowClient(tracking_uri=MLFLOW_URI)
    try:
        exp = client.get_experiment_by_name("nl2sql-finetuning")
        if not exp:
            return None
        runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
        if not runs:
            return None
        run_id = runs[0].info.run_id
        history = client.get_metric_history(run_id, "loss")
        if not history:
            return None

        steps = [m.step for m in history]
        values = [m.value for m in history]

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(steps, values, color="#1565c0", linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Training Loss")
        ax.set_title("Training Loss Curve — Phi-3 mini QLoRA")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        plt.close(fig)
        return buf
    except Exception:
        return None


def build_story(styles: dict, metrics: dict) -> list:
    story = []
    bl = metrics["baseline"]
    ft = metrics["finetuned"]

    # Cover
    story += [
        Spacer(1, 3 * cm),
        Paragraph("NL2SQL Assistant", styles["title"]),
        Paragraph("Full Lifecycle LLM Engineering Project Report", styles["h2"]),
        HRFlowable(width="100%", color=DARK_BLUE, thickness=2),
        Spacer(1, 0.5 * cm),
        Paragraph("Phi-3 mini 3.8B · QLoRA Fine-tuning · RAG Schema Retrieval · MLflow LLMOps", styles["body"]),
        PageBreak(),
    ]

    # Metrics comparison
    story += [Paragraph("Results: Baseline vs Fine-tuned", styles["h1"])]
    em_bl = bl.get("exact_match_accuracy", "N/A")
    em_ft = ft.get("exact_match_accuracy", "N/A")
    ex_bl = bl.get("execution_accuracy", "N/A")
    ex_ft = ft.get("execution_accuracy", "N/A")

    def pct(v):
        return f"{v*100:.1f}%" if isinstance(v, float) else str(v)

    def delta(b, f):
        if isinstance(b, float) and isinstance(f, float):
            return f"+{(f-b)*100:.1f}pp"
        return "N/A"

    metrics_data = [
        ["Metric", "Baseline (zero-shot)", "Fine-tuned + RAG", "Improvement"],
        ["Exact Match Accuracy", pct(em_bl), pct(em_ft), delta(em_bl, em_ft)],
        ["Execution Accuracy", pct(ex_bl), pct(ex_ft), delta(ex_bl, ex_ft)],
        ["Avg Latency (ms)", f"{bl.get('avg_latency_ms', 'N/A'):.0f}" if isinstance(bl.get('avg_latency_ms'), float) else "N/A",
         f"{ft.get('avg_latency_ms', 'N/A'):.0f}" if isinstance(ft.get('avg_latency_ms'), float) else "N/A", "—"],
    ]
    t = Table(metrics_data, colWidths=[5 * cm, 4 * cm, 4 * cm, 3.5 * cm])
    t.setStyle(table_style())
    story += [t, Spacer(1, 0.5 * cm)]

    # Training loss curve
    loss_buf = generate_loss_curve()
    if loss_buf:
        story += [Paragraph("Training Loss Curve", styles["h2"]), Image(loss_buf, width=14 * cm, height=7 * cm)]
    else:
        story += [Paragraph("Training loss curve: run training to populate MLflow.", styles["body"])]

    story += [PageBreak()]

    # Architecture
    story += [
        Paragraph("System Architecture", styles["h1"]),
        Paragraph(
            "The system has three runtime components: (1) <b>Data Pipeline</b> — Spider dataset "
            "download, preprocessing to instruction-tuning JSONL, SQLite schema extraction. "
            "(2) <b>RAG Layer</b> — nomic-embed-text embeddings stored in ChromaDB; at inference, "
            "the top-3 most relevant table schemas are retrieved and injected into the prompt. "
            "(3) <b>Serving Layer</b> — FastAPI backend wraps Ollama; Streamlit provides the UI.",
            styles["body"],
        ),
        Spacer(1, 0.3 * cm),
        Paragraph("Service Launch Commands", styles["h2"]),
    ]

    launch_data = [
        ["Terminal", "Command", "URL"],
        ["1 — Ollama", "ollama serve", "http://localhost:11434"],
        ["2 — MLflow UI", "mlflow ui --port 5000", "http://localhost:5000"],
        ["3 — FastAPI", "uvicorn src.serving.api:app --port 8000 --reload", "http://localhost:8000"],
        ["4 — Streamlit", "streamlit run src/serving/ui.py --server.port 8501", "http://localhost:8501"],
    ]
    t2 = Table(launch_data, colWidths=[3.5 * cm, 8 * cm, 5 * cm])
    t2.setStyle(table_style())
    story.append(t2)

    return story


def generate(output_path: str = "nl2sql_project_report.pdf"):
    styles = get_styles()
    metrics = fetch_mlflow_metrics()

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=2.5 * cm, rightMargin=2.5 * cm,
                            topMargin=2.5 * cm, bottomMargin=2.5 * cm)

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(A4[0] - 2.5 * cm, 1.5 * cm, f"Page {doc.page}")
        canvas.restoreState()

    story = build_story(styles, metrics)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="nl2sql_project_report.pdf")
    args = parser.parse_args()
    generate(args.output)
