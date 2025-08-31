# AI-Powered Natural Language Reporting and Charting Tool

## 📌 Problem Statement

In modern enterprises, decision-makers need **quick, intuitive access to insights** without technical expertise in SQL or BI tools. Traditional reporting requires structured queries and manual chart creation, making it **time-consuming and inaccessible** for non-technical users.

This project solves the problem by enabling **natural language interaction** with enterprise databases. Users can ask questions in plain English (or via voice) and instantly receive **query results, summaries, and interactive charts**.

---

## 🚀 Key Features

* **Natural Language Interface** – Query databases using plain English.
* **AI Query Translation** – Converts natural language into SQL queries automatically.
* **Dynamic Data Retrieval** – Connects to relational databases (MySQL, PostgreSQL, etc.).
* **Automated Visualization** – Generates charts (bar, line, pie, scatter) using Plotly or AI-guided matplotlib.
* **Interactive Results** – Drill down, refine queries, or switch visualization type seamlessly.
* **Export Options** – Download results as CSV, Excel, or PDF.
* **Voice Queries** – Speak your question instead of typing.
* **Query History** – Keep track of past queries and results.

---

## 🏗️ System Architecture

* **Frontend** (`app_gradio.py`)

  * Built with **Gradio**
  * Provides query input, voice input, visualization, and export options
  * Displays AI-generated summaries & query history

* **Backend** (`backend.py`)

  * Built with **FastAPI**
  * Uses **LangChain + OpenAI** to translate NL → SQL
  * Executes queries on MySQL (configurable)
  * Summarizes query results in natural language
  * Optionally generates AI-driven matplotlib charts

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/ai-reporting-tool.git
cd ai-reporting-tool
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

*(Make sure you have MySQL running and update `backend.py` with your DB credentials.)*

### 3. Start Backend (FastAPI)

```bash
python backend.py
```

Backend runs on: `http://localhost:5000`

### 4. Start Frontend (Gradio)

```bash
python app_gradio.py
```

Frontend runs on: `http://localhost:7860`

---

## 🎤 Example Queries

* *"Show me the sales trend for Q2 2025"*
* *"Compare revenue by region for the last 3 years"*
* *"Top 10 clients by trade volume"*
* *"Show me clients who traded options in 2023"*

---

## 📂 Project Structure

```
├── app_gradio.py   # Gradio-based frontend UI
├── backend.py      # FastAPI backend (NLP → SQL + AI summaries + visualization)
├── requirements.txt # Python dependencies
└── README.md       # Documentation
```

---

## ✅ Expected Outcomes

* Faster decision-making with **instant insights**
* Reduced dependency on technical teams for reporting
* **Conversational analytics** experience for business users
* Flexible, scalable solution for enterprise databases

---

## 🔮 Future Enhancements
* Authentication & role-based access
