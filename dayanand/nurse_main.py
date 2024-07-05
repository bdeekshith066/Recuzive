import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

# Set up Google Sheets API credentials
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("path_to_your_credentials.json", scope)
client = gspread.authorize(creds)
sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1lRSSzi6IfIEEKkfpMj67QI3S4ukyKzr2_yDlVEFGQcc/edit?usp=sharing").sheet1

# Function to get all data from Google Sheets
def get_all_data():
    records = sheet.get_all_records()
    return pd.DataFrame(records)

# Function to add a new patient
def add_new_patient(name, age, gender):
    sheet.append_row([name, age, gender, "", "", "", ""])

# Function to get patient data by name
def get_patient_data(name):
    records = get_all_data()
    return records[records['name'] == name]
def app():
    st.title("Patient Management System")

    # Options for nurse to select
    option = st.selectbox("Select an option", ["New Patient", "Existing Patient"])

    if option == "New Patient":
        st.subheader("Add New Patient")
        name = st.text_input("Name")
        age = st.text_input("Age")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        if st.button("Add Patient"):
            add_new_patient(name, age, gender)
            st.success(f"Patient {name} added successfully!")

    elif option == "Existing Patient":
        st.subheader("Retrieve Patient Information")
        name = st.text_input("Enter Patient Name")
        if st.button("Get Patient Info"):
            patient_data = get_patient_data(name)
            if not patient_data.empty:
                st.write(patient_data)
            else:
                st.error("Patient not found!")

    # Display the Google Sheets data in Streamlit
    st.subheader("All Patients Data")
    all_data = get_all_data()
    st.write(all_data)

    
