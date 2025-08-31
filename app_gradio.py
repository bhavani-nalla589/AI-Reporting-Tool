import gradio as gr
import requests
import pandas as pd
import plotly.express as px
import io
from fpdf import FPDF
import speech_recognition as sr
import tempfile
import os

# Backend endpoint
API_URL = "http://localhost:5000/query"

# Store query history
query_history = []


# ---------------------------
# Utility Functions
# ---------------------------
def run_query(nl_query):
    """Send natural language query to backend"""
    try:
        resp = requests.post(API_URL, json={"natural_language_text": nl_query})
        if resp.status_code != 200:
            return None, f"❌ Error {resp.status_code}: {resp.text}", None
        data = resp.json()

        # Save in history
        query_history.append({
            "query": data["query"],
            "nl_query": nl_query,
            "results": data["results"]
        })

        return data, None, query_history
    except Exception as e:
        return None, f"⚠️ Exception: {str(e)}", None


def generate_plot(data, chart_type="Bar"):
    """Generate a Plotly visualization based on chart_type"""
    if not data or not data.get("results"):
        return None

    df = pd.DataFrame(data["results"])
    if df.empty or len(df.columns) < 2:
        return None

    try:
        if chart_type == "Bar":
            fig = px.bar(df, x=df.columns[0], y=df.columns[1])
        elif chart_type == "Line":
            fig = px.line(df, x=df.columns[0], y=df.columns[1])
        elif chart_type == "Pie":
            fig = px.pie(df, names=df.columns[0], values=df.columns[1])
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=df.columns[0], y=df.columns[1])
        else:
            fig = px.bar(df, x=df.columns[0], y=df.columns[1])
        return fig
    except Exception:
        return None


def export_data(results, export_type="csv"):
    """Export results to CSV, Excel, or PDF (returns bytes)"""
    df = pd.DataFrame(results)
    if df.empty:
        return None

    if export_type == "csv":
        return df.to_csv(index=False).encode("utf-8")

    elif export_type == "excel":
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine="openpyxl")
        return buf.getvalue()

    elif export_type == "pdf":
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, txt="AI-Generated Report", ln=True, align="C")

        pdf.set_font("Arial", size=9)
        col_width = max(25, pdf.w / (len(df.columns) + 1))
        row_height = pdf.font_size * 1.6

        # header
        for col in df.columns:
            pdf.cell(col_width, row_height, str(col), border=1)
        pdf.ln(row_height)

        # rows
        for _, row in df.iterrows():
            for item in row:
                pdf.cell(col_width, row_height, str(item), border=1)
            pdf.ln(row_height)

        return pdf.output(dest="S").encode("latin1")

    return None


def refine_query(last_query, refinement_text):
    """Append filter text to last query (NL refinement)"""
    if not query_history:
        return None, "No previous query found", None
    new_query = last_query + " " + refinement_text
    return run_query(new_query)


def voice_to_text(audio_file):
    """Convert speech to text query"""
    if not audio_file or not os.path.exists(audio_file):
        return "No audio provided"
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Speech recognition service error"


# ---------------------------
# Gradio UI
# ---------------------------
with gr.Blocks(css=".gradio-container {background-color: #f9f9f9}") as demo:
    gr.HTML(
        """
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #4A90E2, #9013FE); color: white; border-radius: 12px; margin-bottom: 20px;">
            <h1>🤖 AI POWERED NATURAL LANGUAGE REPORTING AND CHARTING TOOL 📊</h1>
            <p>Ask questions in plain English. Get instant answers, tables, and charts.</p>
        </div>
        """
    )

    with gr.Tab("💬 Query & Results"):
        with gr.Row():
            with gr.Column(scale=3):
                nl_query = gr.Textbox(
                    label="Enter your query in natural language",
                    lines=3,
                    placeholder="e.g., Show me clients who traded in 2023"
                )
                run_btn = gr.Button("🚀 Run Query", variant="primary")

                voice_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="🎤 Speak your query",
                    interactive=True
                )

                refinement_text = gr.Textbox(
                    label="Refine your last query",
                    placeholder="e.g., only for region = New York"
                )
                refine_btn = gr.Button("🔍 Refine Query")

                export_type = gr.Dropdown(
                    ["csv", "excel", "pdf"],
                    value="csv",
                    label="Export Results"
                )
                export_btn = gr.Button("📂 Export Data")
                export_file = gr.File(label="Download Export", visible=True)

            with gr.Column(scale=6):
                chart_type = gr.Dropdown(
                    ["Bar", "Line", "Pie", "Scatter"],
                    value="Bar",
                    label="Chart Type"
                )
                chart_output = gr.Plot(label="📊 Visualization")
                table_output = gr.DataFrame(label="📑 Results Table")
                summary_output = gr.Textbox(label="📝 AI Summary", lines=6)
                error_output = gr.Textbox(label="⚠️ Errors / Logs", lines=4)

    with gr.Tab("📜 Query History"):
        history_output = gr.DataFrame(label="Query History (most recent last)")

    # ---------------------------
    # Event Handlers
    # ---------------------------
    def on_run(nl_text, chart_type):
        data, error, history = run_query(nl_text)
        if error:
            return None, None, None, error, pd.DataFrame(history or [])
        fig = generate_plot(data, chart_type)
        df = pd.DataFrame(data["results"]) if data["results"] else pd.DataFrame()
        history_df = pd.DataFrame(history or [])
        return fig, df, data["natural_language_response"], None, history_df

    def on_refine(ref_text, chart_type):
        if not query_history:
            return None, None, None, "No query to refine", pd.DataFrame()
        last = query_history[-1]["nl_query"]
        data, error, history = refine_query(last, ref_text)
        if error:
            return None, None, None, error, pd.DataFrame(history or [])
        fig = generate_plot(data, chart_type)
        df = pd.DataFrame(data["results"]) if data["results"] else pd.DataFrame()
        history_df = pd.DataFrame(history or [])
        return fig, df, data["natural_language_response"], None, history_df

    def on_export(export_type):
        if not query_history:
            return None
        last_results = query_history[-1]["results"]
        raw = export_data(last_results, export_type)
        if raw is None:
            return None
        ext = "csv" if export_type == "csv" else "xlsx" if export_type == "excel" else "pdf"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
        tmp.write(raw)
        tmp.flush()
        tmp.close()
        return tmp.name

    def on_voice(audio_file, chart_type):
        query_text = voice_to_text(audio_file)
        data, error, history = run_query(query_text)
        if error:
            return None, None, None, error, pd.DataFrame(history or [])
        fig = generate_plot(data, chart_type)
        df = pd.DataFrame(data["results"]) if data["results"] else pd.DataFrame()
        history_df = pd.DataFrame(history or [])
        return fig, df, data["natural_language_response"], None, history_df

    # Bind events
    run_btn.click(
        on_run,
        [nl_query, chart_type],
        [chart_output, table_output, summary_output, error_output, history_output]
    )
    refine_btn.click(
        on_refine,
        [refinement_text, chart_type],
        [chart_output, table_output, summary_output, error_output, history_output]
    )
    export_btn.click(
        on_export,
        [export_type],
        [export_file]
    )
    voice_input.change(
        on_voice,
        [voice_input, chart_type],
        [chart_output, table_output, summary_output, error_output, history_output]
    )


# ---------------------------
# Launch
# ---------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
