import os
import mysql.connector
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import json
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from typing import Dict, List, Any, Optional

# ---------------------------
# CONFIGURATION
# ---------------------------
app = FastAPI()

db_config = {
    'user': 'root',
    'password': 'root',
    'host': 'localhost',
    'database': 'task1'
}

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "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"))

# Cache schema
schema_info = ""


# ---------------------------
# DATABASE SCHEMA EXTRACTION
# ---------------------------
def get_schema_from_database():
    global schema_info
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()

    schema_info = ""
    for table in tables:
        table_name = list(table.values())[0]
        schema_info += f"{table_name}("
        cursor.execute(f"SHOW COLUMNS FROM {table_name}")
        columns = cursor.fetchall()
        col_defs = [column['Field'] for column in columns]
        schema_info += ", ".join(col_defs) + ")\n"

    cursor.close()
    conn.close()


# Load schema on startup
get_schema_from_database()


# ---------------------------
# SQL EXECUTION
# ---------------------------
def execute_query(query: str):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    print("Executing SQL >>>", query)

    try:
        cursor.execute(query)
        results = cursor.fetchall()
    except mysql.connector.Error as err:
        print(f"SQL Error: {err}")
        results = []
    cursor.close()
    conn.close()
    return results


# ---------------------------
# OPENAI HELPERS
# ---------------------------
def natural_language_to_mysql_query(nl_text: str) -> str:
    """Convert natural language request to SQL query using OpenAI"""
    global schema_info
    if not schema_info.strip():
        get_schema_from_database()

    prompt = f"""
    You are a MySQL expert.
    Convert the natural language query into a valid MySQL SELECT statement ONLY.
    Do not include INSERT, UPDATE, DELETE, CREATE, DROP, or explanations.
    
    Schema:
    {schema_info}

    Question:
    {nl_text}

    SQL:
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    sql_query = resp.choices[0].message.content.strip()
    print("LLM Raw Output >>>", sql_query)

    # Remove code fences if present
    if "```" in sql_query:
        sql_query = sql_query.split("```")[1].replace("sql", "").strip()

    # Keep everything from first SELECT onward
    lines = [line.strip() for line in sql_query.splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        if line.lower().startswith("select"):
            sql_query = " ".join(lines[idx:])
            break

    print("Cleaned SQL >>>", sql_query)
    return sql_query


def convert_to_natural_language(data, nl_text: str) -> str:
    """Summarize SQL query results into human-readable text"""
    prompt = f"""
    You are a helpful assistant. 
    Convert the following SQL results into a natural language answer.

    Request: {nl_text}

    Results: {json.dumps(data, indent=2)}

    Answer:
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return resp.choices[0].message.content.strip()


def generate_visualization_with_ai(nl_text: str, query_results: List[Dict], query: str) -> Optional[str]:
    """Generate visualization code via OpenAI and return base64 image"""
    if not query_results:
        return None

    results_str = json.dumps(query_results, indent=2)

    prompt = f"""
    You are a data visualization expert. 
    Based on the user's request, SQL query, and results, 
    suggest the most appropriate matplotlib visualization.

    Request: {nl_text}
    SQL Query: {query}
    Results: {results_str}

    Return only Python code that:
    1. Creates a pandas DataFrame from results
    2. Plots a chart using matplotlib
    3. Saves the figure to a BytesIO buffer and base64 encodes it
    4. Prints ONLY the base64 string

    Do not include explanations.
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    python_code = resp.choices[0].message.content.strip()

    # Extract code block if wrapped
    if "```python" in python_code:
        python_code = python_code.split("```python")[1].split("```")[0].strip()

    exec_globals = {
        'pd': pd,
        'plt': plt,
        'io': io,
        'base64': base64,
        'json': json,
        'results': query_results
    }

    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    try:
        exec(python_code, exec_globals)
    except Exception as e:
        print(f"Visualization error: {e}")
        sys.stdout = old_stdout
        return None

    sys.stdout = old_stdout
    output = mystdout.getvalue().strip()

    return output if output else None


# ---------------------------
# FASTAPI ENDPOINTS
# ---------------------------
class QueryRequest(BaseModel):
    natural_language_text: str


class QueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    natural_language_response: str
    visualization_data: Optional[str] = None


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    nl_text = request.natural_language_text
    sql_query = natural_language_to_mysql_query(nl_text)

    if not sql_query.lower().startswith("select"):
        raise HTTPException(status_code=400, detail=f"Invalid SQL: {sql_query}")

    query_results = execute_query(sql_query)
    natural_language_response = convert_to_natural_language(query_results, nl_text)
    visualization_data = generate_visualization_with_ai(nl_text, query_results, sql_query)

    return {
        "query": sql_query,
        "results": query_results,
        "natural_language_response": natural_language_response,
        "visualization_data": visualization_data
    }


# ---------------------------
# ENTRY POINT
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
