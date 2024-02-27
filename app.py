import pickle
from pathlib import Path

import pandas as pd  # pip install pandas openpyxl
import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit
import streamlit_authenticator as stauth  # pip install streamlit-authenticator
from main import main
from PIL import Image

# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Geniee", page_icon=":cyclone:", layout="wide")


# --- USER AUTHENTICATION ---
names = ["Finsight Demo", "Geniee_admin"]
usernames = ["finsight_demo", "admin"]

# load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    "Geniee", "abcdef", cookie_expiry_days=30)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status:
    st.sidebar.title(f"Welcome {name}")
    #image = Image.open("Logo/LOGO FA.png")
    #st.image(image, width=200)
    main()
    authenticator.logout("Logout", "sidebar")