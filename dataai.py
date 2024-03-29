from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe, SmartDatalake  # Assuming SmartDatalake is part of pandasai
from pandasai.llm import OpenAI
from pandasai.responses.response_parser import ResponseParser
import io
from PIL import Image
from exporttoexc import to_excel
from process_data import load_sheet_data
# Load environment variables from .env file
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')


class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        df=result["value"]
        df_xlsx = to_excel(df)
        st.download_button(label='📥 Download to excel',
                                data=df_xlsx ,
                                file_name= 'df_test.xlsx')
        return

    def format_plot(self, result):
        image = Image.open(result["value"])
        st.image(image)
        pic=result["value"]
        with open(pic, "rb") as file:
            btn = st.download_button(
                    label="Download image",
                    type = 'primary',
                    data=file,
                    file_name=pic,
                    mime="image/png"
                )

        return

    def format_other(self, result):
        st.write(result["value"])
        return

def display_data_ai_session():
    """Display UI for Data-AI session."""
    st.subheader("Data-Geniee")
    data_files = st.sidebar.file_uploader("Upload your data file", type=['csv', 'xlsx'], accept_multiple_files=True)
    if data_files:
        sheet_options = {}
        loaded_data = {}  # Initialize a dictionary to hold loaded data for all files
        for data_file in data_files:
            if data_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" or data_file.type == "application/vnd.ms-excel":
                xls = pd.ExcelFile(data_file)
                for sheet_name in xls.sheet_names:
                    # Unique key for each sheet across all files
                    key = f"{data_file.name} - {sheet_name}"
                    sheet_options[key] = (data_file, sheet_name)
            elif data_file.type == "text/csv":
                # Directly load CSV files without sheet selection
                df = pd.read_csv(data_file)
                key = f"{data_file.name}"
                loaded_data[key] = df

        if sheet_options:
            selected_sheets = st.multiselect("Select sheets to load", options=list(sheet_options.keys()))
            if selected_sheets:
                selected_sheets_info = {key: sheet_options[key] for key in selected_sheets}
                loaded_data.update(load_sheet_data(selected_sheets_info))

        if loaded_data:
            # Processing loaded data with SmartDataframe or SmartDatalake
            if len(loaded_data) > 1:
                # Use SmartDatalake for multiple sheets or files
                sdl = SmartDatalake(list(loaded_data.values()), config={"llm": OpenAI(api_token=api_key),"save_charts": True,"save_charts_path": r"\Charts", "verbose": True, "response_parser": StreamlitResponse})
            else:
                # Use SmartDataframe for a single sheet or file
                sdf = SmartDataframe(next(iter(loaded_data.values())), config={"llm": OpenAI(api_token=api_key),"save_charts": True,"save_charts_path": r"\Charts", "verbose": True, "response_parser": StreamlitResponse})

            # Display data preview for selected sheets or files
            for key, data in loaded_data.items():
                st.write(f"Data Preview for {key}:", data.head())

            user_query = st.text_input("Ask a question about your data")
            if user_query:
                if len(loaded_data) > 1:
                    response = sdl.chat(user_query)
                else:
                    response = sdf.chat(user_query)
        else:
            st.info("Upload one or more files.")
