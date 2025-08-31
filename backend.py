import os
import mysql.connector
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fuzzywuzzy import fuzz
import json
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from datetime import datetime, timedelta
import re
from typing import Dict, List, Any, Optional

app = FastAPI()

# Define database configuration
db_config = {
    'user': 'root',
    'password': 'XXX',
    'host': 'localhost',
    'database': 'task1'
}

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key="OPEN_API_KEY")

# Cache schema information
schema_info = ""
table_names = set()
column_names = set()

def get_schema_from_database():
    global schema_info, table_names, column_names
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()

    schema_info = ""
    table_names = set()
    column_names = set()
    for table in tables:
        table_name = list(table.values())[0]
        table_names.add(table_name.lower())
        schema_info += f"Table: {table_name}\n"
        cursor.execute(f"SHOW COLUMNS FROM {table_name}")
        columns = cursor.fetchall()
        for column in columns:
            column_names.add(column['Field'].lower())
            schema_info += f" - {column['Field']} ({column['Type']})\n"
        # Fetch foreign key relationships
        cursor.execute(f"""
            SELECT
                COLUMN_NAME,
                REFERENCED_TABLE_NAME,
                REFERENCED_COLUMN_NAME
            FROM
                information_schema.KEY_COLUMN_USAGE
            WHERE
                TABLE_NAME = '{table_name}' AND
                REFERENCED_TABLE_NAME IS NOT NULL
        """)
        foreign_keys = cursor.fetchall()
        for fk in foreign_keys:
            schema_info += f" - FK: {fk['COLUMN_NAME']} -> {fk['REFERENCED_TABLE_NAME']}({fk['REFERENCED_COLUMN_NAME']})\n"

    cursor.close()
    conn.close()

# Load schema information when the application starts
get_schema_from_database()

def execute_query(query: str):
    print(f"Executing query: {query}")  # Debugging line
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(query)
        results = cursor.fetchall()
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        results = []
    cursor.close()
    conn.close()
    return results

def convert_to_natural_language(data, natural_language_text: str) -> str:
    if not data:
        return "No data present in the database for the given prompt. Please provide correct data."

    prompt_template = """
    You are a helpful assistant that converts database query results into natural language responses.
    Here is the natural language request:
    {natural_language_text}

    Here are the query results:
    {data}

    Please write a response in natural language based on these results.
    """

    prompt = PromptTemplate(input_variables=["natural_language_text", "data"], template=prompt_template)
    response_chain = LLMChain(prompt=prompt, llm=llm)
    result = response_chain.run(natural_language_text=natural_language_text, data=data)
    return result.strip()

def natural_language_to_mysql_query(natural_language_text: str) -> str:
    global schema_info, table_names, column_names

    # Check for partial matches with fuzzy matching
    def is_match(text, names):
        return any(fuzz.partial_ratio(text.lower(), name) >= 60 for name in names)
   
    if not is_match(natural_language_text, table_names) and not is_match(natural_language_text, column_names):
        return "No data available regarding the given prompt, please provide a correct prompt."

    # Update the prompt to request only the SQL query
    prompt_template = """
    You are a SQL expert. I will provide you with a natural language request and the schema of the database. Please generate the MySQL query to fulfill the request. Provide only the MySQL query, with no additional text or explanation.

    Database Schema:
    {schema_info}

    Natural Language Request:
    {natural_language_text}

    MySQL Query:
    """

    prompt = PromptTemplate(input_variables=["schema_info", "natural_language_text"], template=prompt_template)
    response_chain = LLMChain(prompt=prompt, llm=llm)
    raw_response = response_chain.run(schema_info=schema_info, natural_language_text=natural_language_text)

    # Extract only the SQL query part from the response
    if "MySQL Query:" in raw_response:
        start_index = raw_response.index("MySQL Query:") + len("MySQL Query:")
        sql_query = raw_response[start_index:].strip()
    else:
        sql_query = raw_response.strip()

    return sql_query

def generate_visualization_with_ai(natural_language_text: str, query_results: List[Dict], query: str) -> Optional[str]:
    """
    Use AI to determine and create the most appropriate visualization
    """
    if not query_results:
        return None
       
    # Convert results to a format the AI can understand
    results_str = json.dumps(query_results, indent=2)
   
    prompt_template = """
    You are a data visualization expert. Based on the user's natural language request, the SQL query used, and the query results, determine the most appropriate visualization type and create it using matplotlib.
   
    User's Request: {natural_language_text}
    SQL Query: {query}
    Query Results: {results}
   
    First, analyze what type of visualization would be most appropriate (line chart, bar chart, pie chart, scatter plot, etc.).
    Then, generate Python code to create this visualization using matplotlib. The code should:
    1. Create a pandas DataFrame from the results
    2. Generate an appropriate visualization
    3. Add proper titles, labels, and formatting
    4. Save the figure to a BytesIO buffer and return it as a base64 string
   
    Only return the Python code without any additional explanation. The code should be wrapped in ```python``` blocks.
   
    Example format:
    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import io
    import base64
    import json
   
    # Create DataFrame from results
    data = {results}
    df = pd.DataFrame(data)
   
    # Create visualization
    plt.figure(figsize=(10, 6))
    # ... visualization code here ...
   
    # Save to buffer and convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
   
    print(img_str)
    ```
    """
   
    prompt = PromptTemplate(
        input_variables=["natural_language_text", "query", "results"],
        template=prompt_template
    )
   
    response_chain = LLMChain(prompt=prompt, llm=llm)
   
    try:
        # Get the visualization code from AI
        visualization_code = response_chain.run(
            natural_language_text=natural_language_text,
            query=query,
            results=results_str
        )
       
        # Extract Python code from the response
        if "```python" in visualization_code:
            code_start = visualization_code.index("```python") + len("```python")
            code_end = visualization_code.index("```", code_start)
            python_code = visualization_code[code_start:code_end].strip()
        else:
            python_code = visualization_code.strip()
       
        # Execute the generated code
        exec_globals = {
            'pd': pd,
            'plt': plt,
            'io': io,
            'base64': base64,
            'json': json,
            'results': query_results
        }
       
        # Execute the code and capture the output
        from io import StringIO
        import sys
       
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
       
        exec(python_code, exec_globals)
       
        sys.stdout = old_stdout
        output = mystdout.getvalue().strip()
       
        return output
       
    except Exception as e:
        print(f"Error generating visualization with AI: {e}")
        return None

class QueryRequest(BaseModel):
    natural_language_text: str

class QueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    natural_language_response: str
    visualization_data: Optional[str] = None  # Base64 encoded image

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    nl_text = request.natural_language_text
    mysql_query_or_message = natural_language_to_mysql_query(nl_text)

    if "No data available regarding the given prompt" in mysql_query_or_message:
        raise HTTPException(status_code=400, detail=mysql_query_or_message)
   
    query_results = execute_query(mysql_query_or_message)
    natural_language_response = convert_to_natural_language(query_results, nl_text)
   
    # Generate visualization using AI
    visualization_data = generate_visualization_with_ai(nl_text, query_results, mysql_query_or_message)
   
    return {
        "query": mysql_query_or_message,
        "results": query_results,
        "natural_language_response": natural_language_response,
        "visualization_data": visualization_data
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)