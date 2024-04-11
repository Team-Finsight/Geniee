from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe, SmartDatalake
from pandasai.connectors import MySQLConnector
from pandasai.llm import OpenAI
from pandasai.responses.response_parser import ResponseParser
from io import BytesIO
from PIL import Image
from pandasai import Agent
import sqlalchemy
from sqlalchemy.sql import text
import sqlalchemy
from pandasai import Agent
from pandasai.connectors import MySQLConnector
from Chat_Sugg import chatbot
# Assuming the existence of the to_excel and load_sheet_data functions

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

import json
import os

# Path to the JSON file for storing queries
CACHE_FILE_PATH = 'queries_cache.json'


# Modified function to load queries from the JSON file
def load_queries():
    # Check if the file exists
    if not os.path.exists(CACHE_FILE_PATH):
        # If not, create an empty file with an empty dictionary
        with open(CACHE_FILE_PATH, 'w') as file:
            json.dump({}, file)
        return {}
    else:
        # If it exists, load its content
        with open(CACHE_FILE_PATH, 'r') as file:
            return json.load(file)

def save_query(new_query):
    queries = load_queries()
    # Assuming the new query is unique and using it as a key
    queries[new_query] = new_query
    with open(CACHE_FILE_PATH, 'w') as file:
        json.dump(queries, file)

class StreamlitResponse(ResponseParser):
    def __init__(self, context):
        super().__init__(context)

    def format_dataframe(self, result: dict) -> pd.DataFrame:
        st.dataframe(result["value"])
        df = result["value"]
        df_xlsx = to_excel(df)  # Ensure this function returns a BytesIO object for Excel
        st.download_button(label='📥 Download to excel',
                           data=df_xlsx,
                           file_name='df_test.xlsx')
        return{'type':result['type'],'value':df}

    def format_plot(self, result) -> dict:
        image_paths = result["value"]
        # Iterate over each image path in the list
        for idx, image_path in enumerate(image_paths):
            image = Image.open(image_path)
            st.image(image, caption=f"Image {idx + 1}")

            with open(image_path, "rb") as file:
                st.download_button(label=f"Download Image {idx + 1}",
                                data=file,
                                file_name=f"image_{idx + 1}.png",
                                mime="image/png")
        # Return the original result dict for consistency
        return {'type': result['type'], 'value': image_paths}

    def format_other(self, result):
        if result["type"]=='number' or 'string':
            st.write(result["value"])
        return{'type':result['type'],'value':result['value']}

def to_excel(df):
    """
    Convert a DataFrame into a BytesIO Excel object to be used by Streamlit for downloading.
    This function assumes 'df' is a pandas DataFrame.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.save()
    output.seek(0)
    return output

def load_sheet_data(selected_sheets_info):
    """
    Load data from selected sheets into pandas DataFrames.
    This function assumes 'selected_sheets_info' is a dictionary with keys as sheet names and values as tuples of (file, sheet_name).
    """
    loaded_data = {}
    for key, (data_file, sheet_name) in selected_sheets_info.items():
        df = pd.read_excel(data_file, sheet_name=sheet_name)
        loaded_data[key] = df
    return loaded_data


def update_user_query_from_suggestion(suggestion):
    user_query = suggestion
    

def handle_query_selection(selected_query):
    """Handle actions to be taken after a saved query is selected."""
    # Update the session state with the selected query
    user_query = selected_query

def display_data_ai_session():
    queries = load_queries()
    query_options = list(queries.keys())

    
    st.subheader("Data-Geniee")
    data_source_type = st.sidebar.selectbox("Select your data source", ["File Upload", "SQL Connection"], key="data_source_type")
    loaded_data = {}
    if data_source_type == "File Upload":
        data_files = st.sidebar.file_uploader("Upload your data file", type=['csv', 'xlsx', 'xlsb'], accept_multiple_files=True, key="data_files")
        if data_files:
            sheet_options = {}
            loaded_data = {}
            for data_file in data_files:
                # Check for Excel files (including xlsb)
                if data_file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                                    "application/vnd.ms-excel",
                                    "application/vnd.ms-excel.sheet.binary.macroEnabled.12"]:  # MIME type for .xlsb
                    # Use the appropriate engine for xlsb files
                    engine = 'pyxlsb' if data_file.type == "application/vnd.ms-excel.sheet.binary.macroEnabled.12" else None
                    xls = pd.ExcelFile(data_file, engine=engine)
                    for sheet_name in xls.sheet_names:
                        key = f"{data_file.name} - {sheet_name}"
                        sheet_options[key] = (data_file, sheet_name)
                elif data_file.type == "text/csv":
                    df = pd.read_csv(data_file)
                    key = f"{data_file.name}"
                    loaded_data[key] = df

            if sheet_options:
                selected_sheets = st.multiselect("Select sheets to load", options=list(sheet_options.keys()), key="selected_sheets")
                if selected_sheets:
                    selected_sheets_info = {key: sheet_options[key] for key in selected_sheets}
                    # Function to load sheet data
                    def load_sheet_data(selected_sheets_info):
                        data = {}
                        for key, (file, sheet_name) in selected_sheets_info.items():
                            engine = 'pyxlsb' if file.type == "application/vnd.ms-excel.sheet.binary.macroEnabled.12" else None
                            df = pd.read_excel(file, sheet_name=sheet_name, engine=engine)
                            data[key] = df
                        return data

                    loaded_data.update(load_sheet_data(selected_sheets_info))

            if loaded_data:  # Check if the dictionary is not empty
                for key, data in loaded_data.items():
                    st.write(f"Data Preview for {key}:", data.head())

                user_query = ""

                # Check for direct user input
                user_query_input = st.text_input("Ask a question about your data", key="user_query_input")
                if user_query_input:
                    user_query = user_query_input
                    print("julie ",user_query)
                    if len(loaded_data) == 1:
                        sdf = SmartDataframe(next(iter(loaded_data.values())), config={"llm": OpenAI(api_token=api_key),"save_charts": True,"save_charts_path": r"./Charts", "verbose": True, "response_parser": StreamlitResponse})
                        response = sdf.chat(user_query)     
                    else:
                        sdl = SmartDatalake(list(loaded_data.values()), config={"llm": OpenAI(api_token=api_key),"save_charts": True,"save_charts_path": r"./Charts", "verbose": True, "response_parser": StreamlitResponse})
                        response = sdl.chat(user_query)

                # Button for saving the query - Moved outside the above `if`
                if st.button("Save Query"):
                    save_query(user_query)
                    st.success("Query saved!")
                # Assuming suggestions are generated and displayed as buttons
                if 'suggestions' not in st.session_state or st.session_state.get('regenerate_suggestions', False):
                    first_data_head = next(iter(loaded_data.values())).head()
                    st.session_state['suggestions'] = chatbot(first_data_head)
                    st.session_state['regenerate_suggestions'] = False

                # Display suggestions as buttons
                selected_suggestion = None
                for index, suggestion in enumerate(st.session_state['suggestions']):
                    if st.button(suggestion, key=f"suggestion_{index}"):
                        selected_suggestion = suggestion
                        break  # Exit loop if a suggestion is selected

                # Handle the case when a suggestion is selected
                if selected_suggestion:
                    user_query = selected_suggestion
                    print("julie ",user_query)
                    # Separate section or UI element for saving the selected query
                    if len(loaded_data) == 1:
                        sdf = SmartDataframe(next(iter(loaded_data.values())), config={"llm": OpenAI(api_token=api_key),"save_charts": True,"save_charts_path": r"./Charts", "verbose": True, "response_parser": StreamlitResponse})
                        response = sdf.chat(user_query)     
                    else:
                        sdl = SmartDatalake(list(loaded_data.values()), config={"llm": OpenAI(api_token=api_key),"save_charts": True,"save_charts_path": r"./Charts", "verbose": True, "response_parser": StreamlitResponse})
                        response = sdl.chat(user_query)
                # Check for selection from saved queries
                query_options = load_queries()  # Assuming returns a dictionary of saved queries
                selected_query = st.sidebar.multiselect("Select a saved query", options=list(query_options.keys()))
                if selected_query:
                    user_query = query_options[selected_query[0]]
                    print("julie ",user_query)
                    if len(loaded_data) == 1:
                        sdf = SmartDataframe(next(iter(loaded_data.values())), config={"llm": OpenAI(api_token=api_key),"save_charts": True,"save_charts_path": r"./Charts", "verbose": True, "response_parser": StreamlitResponse})
                        response = sdf.chat(user_query)     
                    else:
                        sdl = SmartDatalake(list(loaded_data.values()), config={"llm": OpenAI(api_token=api_key),"save_charts": True,"save_charts_path": r"./Charts", "verbose": True, "response_parser": StreamlitResponse})
                        response = sdl.chat(user_query)
                
        else:
            st.info("Upload one or more files.")

    elif data_source_type == "SQL Connection":
        st.sidebar.text_input("Host", key="host")
        st.sidebar.number_input("Port", min_value=0, max_value=65535, value=3306, key="port")
        st.sidebar.text_input("Database", key="database")
        st.sidebar.text_input("Username", key="username")
        st.sidebar.text_input("Password", type="password", key="password")
        st.sidebar.text_input("Table", key="table")
        condition_column = st.sidebar.text_input("Condition Column (Optional)", key="condition_column")
        condition_value = st.sidebar.text_input("Condition Value (Optional)", key="condition_value")

                # Fetch and process data based on the user query
        user_query = st.text_input("Ask a question about your data")
        suggestions= chatbot(data.head())
        st.markdown("### Suggestions for questions to ask:")
        st.info(suggestions)
        if user_query:
            # SQL connection settings
            host = st.session_state.host
            port = st.session_state.port
            database = st.session_state.database
            username = st.session_state.username
            password = st.session_state.password
            table = st.session_state.table

            # Initialize your MySQL connector here
            sql_connector = MySQLConnector(
                config={
                    "host": host,
                    "port": port,
                    "database": database,
                    "username": username,
                    "password": password,
                    "table": table,
                    "where": [[condition_column, "=", condition_value]] if condition_column and condition_value else [],
                }
            )

            # Initialize the Agent with your SQL connector and configuration
            agent = Agent([sql_connector], config={"llm": OpenAI(api_token=api_key), "direct_sql": True, "save_charts": True,"save_charts_path": r"./Charts", "verbose": True, "response_parser": StreamlitResponse})

            # Use the agent to process the user query
            response = agent.chat(user_query)
            st.write(response)
        else:
            st.info("Please enter your query to process the data.")