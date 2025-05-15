import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

from streamlit_option_menu import option_menu
USER_FILE="siguodata.csv"

if not os.path.exists(USER_FILE):
    with open(USER_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["email","password"])



# username=["ajay@gmail.com","ajay123@gmail.com"]
# userpassword=["123","1234"]

if "austhonticade" not in st.session_state:
    st.session_state.authonticated=False
if "user_id" not in st.session_state:
     st.session_state.user_id=None

query_parms=st.query_params

if "auth" in query_parms and query_parms["auth"]=="true":
    st.session_state.authonticated=True
    st.session_state.user_id=query_parms.get("user","")


with st.sidebar:
    if st.session_state.authonticated:
        selected=option_menu("dataset",["dataset","logout"])

    else:
        selected=option_menu("Logout",["login","signup"])



if selected=="login":
    with st.form(key="form1"):
        email=st.text_input("Enetr your email")
        passward=st.text_input("Enter ayour password",type="password")
        submit=st.form_submit_button(label="submit")
        if submit:
            with open(USER_FILE, mode='r') as file:
                users = list(csv.reader(file))

            if [email, passward] in users:
                st.success("Login successful!")
                st.session_state.authenticated = True
                st.session_state.user_id = email
                st.success("login success")
                st.session_state.authonticated = True
                st.session_state.user_id = email
                st.query_params.update(auth="true", user=email)
                st.rerun()
            else:
                st.error("Invalid credentials")



if selected=="signup":
    with st.form(key="signup_form"):
        email = st.text_input("Enter your email")
        password = st.text_input("Enter your password", type="password")
        submit = st.form_submit_button(label="Sign Up")

        if submit:
            with open(USER_FILE, mode='r') as file:
                users = list(csv.reader(file))

            if any(len(row) > 0 and row[0] == email for row in users[1:]):
                st.error("Email already registered!")
            else:
                with open(USER_FILE, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([email, password])
                st.success("Signup successful! You can now login.")


if selected == "logout":
    st.session_state.authonticated = False
    st.session_state.user_id = None
    st.query_params.clear()
    st.rerun()

if selected=="dataset":
    st.header("session creat")
    st.write("session id",st.session_state.user_id)

#
# import streamlit as st
#
# from streamlit_option_menu import option_menu
#
# import pandas as pd
#
#
#
# useremails = ["abc@gmail.com","admin@gmail.com","twinkle@gmail.com"]
#
# userpasswords = ["1234","2345","8141"]
#
#
# if "authenticated" not in st.session_state:
#
#   st.session_state.authenticated = False
#
# if "user_id" not in st.session_state:
#
#   st.session_state.user_id = None
#
#
# query_params = st.query_params
#
# if "auth" in query_params and query_params["auth"] == "true":
#
#   st.session_state.authenticated = True
#
#   st.session_state.user_id = query_params.get("user", "")
#
#
# with st.sidebar:
#
#    if st.session_state.authenticated:
#
#        selected = option_menu("Admin Panel",["Dashboard","About us","Logout","Dataset","Setting"],
#
#                               menu_icon=["house"],default_index=0,orientation="vertcal",
#
#                               icons=["cast","people","lock","table","gear"])
#
#    else:
#
#        selected = option_menu("Admin Panel", ["Dashboard", "About us", "Login", "Dataset", "Setting"],
#
#                               menu_icon=["house"], default_index=0, orientation="vertcal",
#
#                               icons=["cast", "people", "lock", "table", "gear"])
#
#
#
# if selected=="Dashboard":
#
#    st.header("Dashboard")
#
#    st.write("Project Details",st.session_state.user_id)
#
#
# if selected=="About us":
#
#    st.header("About us")
#
#    st.write("this is my project build using AI/ML")
#
#
# if selected=="Login":
#
#    st.header("Login")
#
#    with st.form(key="form1"):
#
#        email = st.text_input("Enter Email")
#
#        password = st.text_input("Enter Password",type="password")
#
#        submitbtn = st.form_submit_button(label="login")
#
#
#        if submitbtn:
#
#            if email in useremails:
#
#                index = useremails.index(email)
#
#                if password==userpasswords[index]:
#
#                    # if correct email or password
#
#                    st.session_state.authenticated = True
#
#                    st.session_state.user_id = email
#
#                    st.query_params.update(auth="true", user=email)
#
#                    st.success("Login Successfully")
#
#                    st.rerun()
#
#                else:
#
#                    st.error("Invalid email or passwords")
#
#
#
#
#
# if selected=="Dataset":
#
#    st.header("Dataset of our project")
#
#    df = pd.read_csv("RESULT1.xlsx")
#
#    st.dataframe(df)
#
#
# if selected=="Setting":
#
#    st.header("Settings")
#
#    st.write("Change Password")
#
#
# if selected == "Logout":
#
#       st.session_state.authenticated = False
#
#       st.session_state.user_id = None
#
#       st.query_params.clear()  # Clear query params on logout
#
#       st.success("Logged out successfully!")
#
#       st.rerun()
#
