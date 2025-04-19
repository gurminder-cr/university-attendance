import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
import pickle
import time
import pandas as pd
from PIL import Image
from datetime import datetime
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import base64

#bg-image



with open('img-1.jpg','rb') as f:
    data=f.read()
imgs= base64.b64encode(data).decode()

css = f"""
    <style>
    /* Set background image */
    [data-testid="stAppViewContainer"] {{
        background-image: url('data:image/png;base64,{imgs}');
        background-size: cover;
        background-position: center; /* Ensures proper alignment */
    }}

    /* Customize button color */
    [data-testid="stButton"] {{
        color: yellow;
    }}

    /* Stylish header */
    .register-header {{
        font-size: 32px; /* Slightly larger for better visibility */
        font-weight: bold;
        color: #fff;
        text-align: center;
        background: linear-gradient(90deg, #ff7b00, #ff0080); /* More natural gradient */
        padding: 15px 20px; /* Balanced padding */
        border-radius: 12px; /* Slightly smoother edges */
        box-shadow: 0px 5px 12px rgba(0, 0, 0, 0.3); /* Deeper shadow for a subtle 3D effect */
        width: 80%; /* Adjusts width to fit content */
        margin: 20px auto; /* Centers the header properly */
        display: inline-block; /* Prevents full width stretching */
    }}
    </style>
"""


st.markdown(css, unsafe_allow_html=True)
# Configure Streamlit layout
# st.set_page_config(page_title="University Attendance System", page_icon="📚",layout="wide", initial_sidebar_state="expanded")


# Define file paths
registered_faces_folder = "registered_students"
encoding_file = "face_encodings.pkl"
attendance_file = "attendance.csv"

# Ensure required directories exist
if not os.path.exists(registered_faces_folder):
    os.makedirs(registered_faces_folder)

if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["Name", "Roll Number", "Department", "Year", "Date", "Timestamp"]).to_csv(attendance_file, index=False)

# Load face encodings
registered_face_encodings = []
registered_face_names = []


def save_encodings():
    with open(encoding_file, "wb") as f:
        pickle.dump((registered_face_encodings, registered_face_names), f)

def load_encodings():
    global registered_face_encodings, registered_face_names
    if os.path.exists(encoding_file):
        with open(encoding_file, "rb") as f:
            registered_face_encodings, registered_face_names = pickle.load(f)

load_encodings()

# Image Enhancement
def enhance_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.GaussianBlur(image, (5, 5), 0)

# st.image("logo-512x512-1.png", width=180) 
# Create three columns
col1, col2, col3 = st.columns([1.5, 2, 1])  # Adjust ratios for spacing

# Place the logo in the center column
with col2:
    st.image("logo-512x512-1.png", width=200) 
    

# Register New Student

def register_face():
    st.markdown(
    """
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">
        <h2 style="text-align: center;">📝 Register Student</h2>
    </div>
    """,
    unsafe_allow_html=True
)
    if lottie_reg:
        st_lottie(lottie_reg, speed=1, height=200, key="anime")
    else:
        st.error("Failed to load animation")
    
    name = st.text_input("Student Name", placeholder="Enter full name")
    roll_no = st.text_input("Roll Number", placeholder="Enter roll number")
    department = st.text_input("Department", placeholder="Enter department name")
    year = st.selectbox("Year", ["1st", "2nd", "3rd", "4th"])
    image = st.file_uploader("Upload Image", type=['jpg', 'png'], help="Upload a clear image with one face visible")

    if st.button("Register") and image is not None and name and roll_no and department and year:
        img = np.array(Image.open(image))
        rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        face_locations = face_recognition.face_locations(rgb_img)
        if len(face_locations) != 1:
            st.error("⚠️ Please upload an image with exactly one face.")
            return

        if any(student['roll_no'] == roll_no for student in registered_face_names):
            st.error("⚠️ Student with this Roll Number is already registered!")
            return

        file_name = f"{name}_{roll_no}_{department}_{year}.jpg"
        file_path = os.path.abspath(os.path.join(registered_faces_folder, file_name))  # Ensure correct path
        st.write(f"Saving Image at: {file_path}")  # Debugging

        enhanced_face = enhance_image(rgb_img)
        cv2.imwrite(file_path, enhanced_face)

        encoding = face_recognition.face_encodings(rgb_img, [face_locations[0]])[0]
        registered_face_encodings.append(encoding)
        registered_face_names.append({'name': name, 'roll_no': roll_no, 'department': department, 'year': year, 'file': file_name})

        save_encodings()
        if lottie_reg:
            st_lottie(lottie_ver, speed=1, height=200)
        else:
            st.error("Failed to load animation")
        st.success("✅ Student Registered Successfully!")
        


# View Registered Faces
def view_registered_faces():
    st.header("📂 Registered Students")
    search_query = st.text_input("🔍 Search by Name or Roll Number")
    cols = st.columns(5)

    filtered_faces = [
        face for face in registered_face_names if 
        search_query.lower() in face['name'].lower() or search_query in face['roll_no']
    ] if search_query else registered_face_names

    if not filtered_faces:
        st.warning("⚠️ No matching students found!")

    for idx, face in enumerate(filtered_faces):
        file_path = os.path.abspath(os.path.join(registered_faces_folder, face['file']))
        st.write(f"Debug Path: {file_path}")  # Debugging

        with cols[idx % 5]:
            if os.path.exists(file_path):
                st.image(file_path, caption=f"{face['name']} ({face['roll_no']})", width=100)
            else:
                st.error(f"⚠️ Image file not found: {file_path}")

# Mark Attendance
# def mark_attendance(name, roll_no, department, year):
#     today = datetime.today().strftime('%Y-%m-%d')
#     timestamp = time.strftime('%H:%M:%S')

#     df = pd.read_csv(attendance_file)

#     if not ((df["Roll Number"] == roll_no) & (df["Date"] == today)).any():
#         new_entry = pd.DataFrame([[name, roll_no, department, year, today, timestamp]],
#                                  columns=["Name", "Roll Number", "Department", "Year", "Date", "Timestamp"])
#         df = pd.concat([df, new_entry], ignore_index=True)
#         df.to_csv(attendance_file, index=False)
#         return True
#     return False

def mark_attendance(name, roll_no, department, year):
    today = datetime.today().strftime('%Y-%m-%d')
    timestamp = time.strftime('%H:%M:%S')

    df = pd.read_csv(attendance_file)

    # Convert roll_no to string to match CSV data
    roll_no = str(roll_no)

    # Check if the student has already been marked today
    if not ((df["Roll Number"].astype(str) == roll_no) & (df["Date"] == today)).any():
        new_entry = pd.DataFrame([[name, roll_no, department, year, today, timestamp]],
                                 columns=["Name", "Roll Number", "Department", "Year", "Date", "Timestamp"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(attendance_file, index=False)
        return True  # Attendance marked successfully
    return False  # Already marked today

# Recognize Face in Live Camera
def recognize_live_camera():
    st.header("🎥 Live Face Recognition")
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    details_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Camera not working!")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        detected_faces = []

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            face_distances = face_recognition.face_distance(registered_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < 0.5:
                matched_person = registered_face_names[best_match_index]
                name, roll_no, department, year = matched_person['name'], matched_person['roll_no'], matched_person['department'], matched_person['year']
                
                if mark_attendance(name, roll_no, department, year):
                    details = f"✅ Attendance Marked\n📌 Roll No: {roll_no}\n📚 Dept: {department}\n📅 Year: {year}"
                else:
                    details = f"⚠️ Already Marked Today\n📌 Roll No: {roll_no}\n📚 Dept: {department}\n📅 Year: {year}"
            else:
                name, details = "Unknown", "Unknown"

            cv2.rectangle(frame, (left*2, top*2), (right*2, bottom*2), (0, 255, 0), 2)
            cv2.putText(frame, name, (left*2, top*2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            detected_faces.append(f"**{name}**\n{details}")

        frame_placeholder.image(frame, channels="BGR")
        details_placeholder.markdown("\n\n".join(detected_faces))

    cap.release()
    cv2.destroyAllWindows()

# Export Attendance
def export_attendance():
    st.header("📤 Export Attendance")
    df=pd.read_csv(attendance_file)
    st.write(df)
    if os.path.exists(attendance_file):
        with open(attendance_file, "rb") as f:
            st.download_button(label="📥 Download Attendance", data=f, file_name="attendance.csv", mime="text/csv")


#lottie
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_reg= load_lottieurl("https://lottie.host/c5b7ad1a-5aab-40f7-9e08-a06073f20c56/vB8OIINhEE.json")
lottie_ver=load_lottieurl("https://lottie.host/4c6b7e86-92b9-499b-b74d-c36b82607569/DvBgPGE964.json")
lottie_rec= load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_hq6m4wvk.json")
lottie_view= load_lottieurl("https://lottie.host/65b0b2c5-cdbd-4a1a-838b-4d3d0f5955a2/qAx3DOW9Mm.json")
lottie_att= load_lottieurl("https://lottie.host/8e9d7241-637b-486a-bdf8-4e16293d2753/e4w8reRCDp.json")

# Main Function
def main():
    with st.sidebar:
        menu = option_menu("Choose an Option", ["Register Student", "Recognize Attendance", "View Registered", "Export Attendance"], menu_icon="cast", default_index=0)

    if menu == "Register Student":
        
        register_face()
    elif menu == "Recognize Attendance":
        if lottie_reg:
            st_lottie(lottie_reg, speed=1, width=500, height=200, key="anime")
        else:
            st.error("Failed to load animation")
        recognize_live_camera()
    elif menu == "View Registered":
        if lottie_view:
            st_lottie(lottie_view, speed=1, height=150, key="anime")
        else:
            st.error("Failed to load animation")
        view_registered_faces()
    elif menu == "Export Attendance":
        
        export_attendance()
        if lottie_reg:
            st_lottie(lottie_att, speed=1, height=400)
        else:
            st.error("Failed to load animation")

if __name__ == '__main__':
    main()
