import streamlit as st
import cv2
import face_recognition
import numpy as np
from PIL import Image
import os
import imutils
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

registered_faces_folder = "registered_students"
if not os.path.exists(registered_faces_folder):
    os.makedirs(registered_faces_folder)

registered_face_encodings = []
registered_face_names = []

def load_registered_faces():
    global registered_face_encodings, registered_face_names
    registered_face_encodings.clear()
    registered_face_names.clear()
    
    for file_name in os.listdir(registered_faces_folder):
        if file_name.endswith(".jpg"):
            parts = file_name.replace(".jpg", "").split("_")
            if len(parts) == 4:
                name, roll_no, department, year = parts
                img_path = os.path.join(registered_faces_folder, file_name)
                img = face_recognition.load_image_file(img_path)
                face_locations = face_recognition.face_locations(img)
                if face_locations:
                    encoding = face_recognition.face_encodings(img, face_locations)[0]
                    registered_face_encodings.append(encoding)
                    registered_face_names.append({
                        'name': name, 'roll_no': roll_no, 'department': department, 'year': year, 'file': file_name
                    })

load_registered_faces()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def enhance_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = imutils.resize(image, width=400)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    return image

def register_face():
    st.header("📝 Register Student")
    name = st.text_input("Student Name")
    roll_no = st.text_input("Roll Number")
    department = st.text_input("Department")
    year = st.selectbox("Year", ["1st", "2nd", "3rd", "4th"])
    image = st.file_uploader("Upload Image", type=['jpg', 'png'])
    
    if st.button("Register") and image is not None and name and roll_no and department and year:
        img = np.array(Image.open(image))
        faces = face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 1.3, 5)
        if len(faces) == 1:
            x, y, w, h = faces[0]
            cropped_face = img[y:y+h, x:x+w]
            enhanced_face = enhance_image(cropped_face)
            file_path = os.path.join(registered_faces_folder, f"{name}_{roll_no}_{department}_{year}.jpg")
            cv2.imwrite(file_path, enhanced_face)
            st.success("✅ Student Registered Successfully!")
            load_registered_faces()
        else:
            st.error("⚠️ Please upload an image with exactly one face.")

def view_registered_faces():
    st.header("📂 Registered Students")
    cols = st.columns(5)
    for idx, face in enumerate(registered_face_names):
        with cols[idx % 5]:
            st.image(
                os.path.join(registered_faces_folder, face['file']),
                caption=f"{face['name']}\n{face['roll_no']}",
                width=100
            )

def recognize_live_camera():
    st.header("🎥 Live Face Recognition")
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    details_placeholder = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not working!")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        detected_faces = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(registered_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"
            details = "Unknown"
            
            if True in matches:
                match_index = matches.index(True)
                matched_person = registered_face_names[match_index]
                name = matched_person['name']
                details = f"Roll No: {matched_person['roll_no']}\nDepartment: {matched_person['department']}\nYear: {matched_person['year']}"
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            detected_faces.append(f"**{name}**\n{details}")
        
        frame_placeholder.image(frame, channels="BGR")
        details_placeholder.markdown("\n\n".join(detected_faces))
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    with st.sidebar:
        st.title("📌 Menu")
        menu = option_menu(
            menu_title="Choose an Option",
            options=["Register Student", "Recognize Attendance", "View Registered", "About"],
            icons=["person-plus", "camera", "list-task", "info-circle"],
            menu_icon="cast",
            default_index=0,
            orientation="vertical"
        )

    st.title("🎓 University Attendance System")
    st.markdown("---")

    if menu == "Register Student":
        register_face()
    elif menu == "Recognize Attendance":
        recognize_live_camera()
    elif menu == "View Registered":
        view_registered_faces()
    elif menu == "About":
        st.markdown("""
        - **University Attendance System**
        - Uses advanced Face Recognition technology.
        - Secure, efficient, and easy-to-use.
        - Developed using **Streamlit** and **OpenCV**.
        """)

if __name__ == '__main__':
    main()
