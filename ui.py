# import streamlit as st
# import cv2
# import face_recognition
# import numpy as np
# from PIL import Image
# import os
# import time
# import imutils
# from streamlit_option_menu import option_menu

# st.set_page_config(layout="wide", initial_sidebar_state="expanded")
# # st.title("📚 University Attendance System")

# registered_faces_folder = "registered_students"
# if not os.path.exists(registered_faces_folder):
#     os.makedirs(registered_faces_folder)

# registered_face_encodings = []
# registered_face_names = []

# # Load registered faces
# def load_registered_faces():
#     global registered_face_encodings, registered_face_names
#     registered_face_encodings.clear()
#     registered_face_names.clear()
    
#     for file_name in os.listdir(registered_faces_folder):
#         if file_name.endswith(".jpg"):
#             name, roll_no = file_name.replace(".jpg", "").split("_")[:2]
#             img_path = os.path.join(registered_faces_folder, file_name)
#             img = face_recognition.load_image_file(img_path)
#             face_locations = face_recognition.face_locations(img)
#             if face_locations:
#                 encoding = face_recognition.face_encodings(img, face_locations)[0]
#                 registered_face_encodings.append(encoding)
#                 registered_face_names.append({'name': name, 'roll_no': roll_no, 'file': file_name})

# load_registered_faces()

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# def enhance_image(image):
#     """Enhance face quality using sharpening and brightness adjustment."""
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     image = imutils.resize(image, width=400)
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#     image = cv2.filter2D(image, -1, kernel)
#     return image

# def register_face():
#     st.header("📝 Register Student")
#     name = st.text_input("Student Name")
#     roll_no = st.text_input("Roll Number")
#     image = st.file_uploader("Upload Image", type=['jpg', 'png'])
    
#     if st.button("📸 Capture from Live Camera"):
#         cap = cv2.VideoCapture(0)
#         st.write("Press 's' to Capture & Save, 'q' to Exit")
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             cv2.imshow("Camera", frame)
#             if cv2.waitKey(1) & 0xFF == ord('s'):
#                 image = frame
#                 cap.release()
#                 cv2.destroyAllWindows()
#                 break
        
#     if st.button("Register") and image is not None:
#         img = np.array(Image.open(image)) if isinstance(image, Image.Image) else image
#         faces = face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 1.3, 5)
#         if len(faces) == 1:
#             x, y, w, h = faces[0]
#             cropped_face = img[y:y+h, x:x+w]
#             enhanced_face = enhance_image(cropped_face)
#             file_path = os.path.join(registered_faces_folder, f"{name}_{roll_no}.jpg")
#             cv2.imwrite(file_path, enhanced_face)
#             st.success("✅ Student Registered Successfully!")
#             load_registered_faces()
#         else:
#             st.error("⚠️ Please upload an image with exactly one face.")

# def recognize_live_camera():
#     video_capture = cv2.VideoCapture(0)
#     if not video_capture.isOpened():
#         st.error("Unable to access the camera.")
#         return

#     st.write("Press 'q' to stop the camera.")
#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             continue
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         face_locations = face_recognition.face_locations(rgb_frame)
#         face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
#         for face_encoding, face_location in zip(face_encodings, face_locations):
#             matches = face_recognition.compare_faces(registered_face_encodings, face_encoding)
#             face_distances = face_recognition.face_distance(registered_face_encodings, face_encoding)
#             name = "Unknown"
#             if True in matches:
#                 best_match_index = np.argmin(face_distances)
#                 name = registered_face_names[best_match_index]['name']
            
#             top, right, bottom, left = face_location
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#             cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
#         cv2.imshow("Live Camera", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     video_capture.release()
#     cv2.destroyAllWindows()
#     st.success("📷 Camera Stopped Successfully.")

# def view_registered_faces():
#     st.header("📂 Registered Students")
#     for face in registered_face_names:
#         st.image(os.path.join(registered_faces_folder, face['file']), caption=f"Name: {face['name']}, Roll No: {face['roll_no']}", use_column_width=True)

# def main():
   

#     # Sidebar Option Menu with Icons
#     with st.sidebar:
#         st.title("📌 Menu")
#         menu = option_menu(
#             menu_title="Choose an Option",
#             options=["Register Student", "Recognize Attendance", "View Registered", "About"],
#             icons=["person-plus", "camera", "list-task", "info-circle"],
#             menu_icon="cast",
#             default_index=0,
#             orientation="vertical"
#         )

#     # Page Content Based on Selected Option
#     st.title("🎓 University Attendance System")
#     st.markdown("---")

#     if menu == "Register Student":
#         st.subheader("📄 Register New Student")
#         st.write("Fill in student details and register their face for attendance recognition.")
#         register_face()

#     elif menu == "Recognize Attendance":
#         st.subheader("🎥 Recognize Attendance")
#         st.write("Start live camera to automatically mark attendance using face recognition.")
#         if st.button("▶️ Start Camera"):
#             recognize_live_camera()

#     elif menu == "View Registered":
#         st.subheader("📋 View Registered Students")
#         st.write("List of all students registered in the system with face IDs.")
#         view_registered_faces()

#     elif menu == "About":
#         st.subheader("ℹ️ About System")
#         st.markdown("""
#         - **University Attendance System**
#         - Uses advanced Face Recognition technology.
#         - Secure, efficient, and easy-to-use.
#         - Supports live camera registration and face recognition.
#         - Developed using **Streamlit** and **OpenCV**.
#         """)
# if __name__ == '__main__':
#     main()


# import streamlit as st
# import cv2
# import face_recognition
# import numpy as np
# from PIL import Image
# import os
# import imutils
# from streamlit_option_menu import option_menu

# st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# registered_faces_folder = "registered_students"
# if not os.path.exists(registered_faces_folder):
#     os.makedirs(registered_faces_folder)

# registered_face_encodings = []
# registered_face_names = []

# # Load registered faces

# def load_registered_faces():
#     global registered_face_encodings, registered_face_names
#     registered_face_encodings.clear()
#     registered_face_names.clear()
    
#     for file_name in os.listdir(registered_faces_folder):
#         if file_name.endswith(".jpg"):
#             parts = file_name.replace(".jpg", "").split("_")
#             if len(parts) == 4:
#                 name, roll_no, department, year = parts
#                 img_path = os.path.join(registered_faces_folder, file_name)
#                 img = face_recognition.load_image_file(img_path)
#                 face_locations = face_recognition.face_locations(img)
#                 if face_locations:
#                     encoding = face_recognition.face_encodings(img, face_locations)[0]
#                     registered_face_encodings.append(encoding)
#                     registered_face_names.append({'name': name, 'roll_no': roll_no, 'department': department, 'year': year, 'file': file_name})

# load_registered_faces()

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# def enhance_image(image):
#     """Enhance face quality using sharpening and brightness adjustment."""
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     image = imutils.resize(image, width=400)
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#     image = cv2.filter2D(image, -1, kernel)
#     return image

# def register_face():
#     st.header("📝 Register Student")
#     name = st.text_input("Student Name")
#     roll_no = st.text_input("Roll Number")
#     department = st.text_input("Department")
#     year = st.selectbox("Year", ["1st", "2nd", "3rd", "4th"])
#     image = st.file_uploader("Upload Image", type=['jpg', 'png'])
    
#     if st.button("Register") and image is not None and name and roll_no and department and year:
#         img = np.array(Image.open(image))
#         faces = face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 1.3, 5)
#         if len(faces) == 1:
#             x, y, w, h = faces[0]
#             cropped_face = img[y:y+h, x:x+w]
#             enhanced_face = enhance_image(cropped_face)
#             file_path = os.path.join(registered_faces_folder, f"{name}_{roll_no}_{department}_{year}.jpg")
#             cv2.imwrite(file_path, enhanced_face)
#             st.success("✅ Student Registered Successfully!")
#             load_registered_faces()
#         else:
#             st.error("⚠️ Please upload an image with exactly one face.")

# def recognize_live_camera():
#     st.error("Live camera capture has been removed.")

# def view_registered_faces():
#     st.header("📂 Registered Students")
#     for face in registered_face_names:
#         st.image(os.path.join(registered_faces_folder, face['file']), caption=f"Name: {face['name']}, Roll No: {face['roll_no']}, Department: {face['department']}, Year: {face['year']}", use_column_width=True)

# def main():
#     with st.sidebar:
#         st.title("📌 Menu")
#         menu = option_menu(
#             menu_title="Choose an Option",
#             options=["Register Student", "Recognize Attendance", "View Registered", "About"],
#             icons=["person-plus", "camera", "list-task", "info-circle"],
#             menu_icon="cast",
#             default_index=0,
#             orientation="vertical"
#         )

#     st.title("🎓 University Attendance System")
#     st.markdown("---")

#     if menu == "Register Student":
#         st.subheader("📄 Register New Student")
#         st.write("Fill in student details and register their face for attendance recognition.")
#         register_face()

#     elif menu == "Recognize Attendance":
#         st.subheader("🎥 Recognize Attendance")
#         st.write("Start live camera to automatically mark attendance using face recognition.")
#         recognize_live_camera()

#     elif menu == "View Registered":
#         st.subheader("📋 View Registered Students")
#         st.write("List of all students registered in the system with face IDs.")
#         view_registered_faces()

#     elif menu == "About":
#         st.subheader("ℹ️ About System")
#         st.markdown("""
#         - **University Attendance System**
#         - Uses advanced Face Recognition technology.
#         - Secure, efficient, and easy-to-use.
#         - Developed using **Streamlit** and **OpenCV**.
#         """)

# if __name__ == '__main__':
#     main()

# import streamlit as st
# import cv2
# import face_recognition
# import numpy as np
# from PIL import Image
# import os
# import imutils
# from streamlit_option_menu import option_menu

# st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# registered_faces_folder = "registered_students"
# if not os.path.exists(registered_faces_folder):
#     os.makedirs(registered_faces_folder)

# registered_face_encodings = []
# registered_face_names = []

# def load_registered_faces():
#     global registered_face_encodings, registered_face_names
#     registered_face_encodings.clear()
#     registered_face_names.clear()
    
#     for file_name in os.listdir(registered_faces_folder):
#         if file_name.endswith(".jpg"):
#             parts = file_name.replace(".jpg", "").split("_")
#             if len(parts) == 4:
#                 name, roll_no, department, year = parts
#                 img_path = os.path.join(registered_faces_folder, file_name)
#                 img = face_recognition.load_image_file(img_path)
#                 face_locations = face_recognition.face_locations(img)
#                 if face_locations:
#                     encoding = face_recognition.face_encodings(img, face_locations)[0]
#                     registered_face_encodings.append(encoding)
#                     registered_face_names.append({
#                         'name': name, 'roll_no': roll_no, 'department': department, 'year': year, 'file': file_name
#                     })

# load_registered_faces()

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# def enhance_image(image):
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     image = imutils.resize(image, width=400)
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#     image = cv2.filter2D(image, -1, kernel)
#     return image

# def register_face():
#     st.header("📝 Register Student")
#     name = st.text_input("Student Name")
#     roll_no = st.text_input("Roll Number")
#     department = st.text_input("Department")
#     year = st.selectbox("Year", ["1st", "2nd", "3rd", "4th"])
#     image = st.file_uploader("Upload Image", type=['jpg', 'png'])
    
#     if st.button("Register") and image is not None and name and roll_no and department and year:
#         img = np.array(Image.open(image))
#         faces = face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 1.3, 5)
#         if len(faces) == 1:
#             x, y, w, h = faces[0]
#             cropped_face = img[y:y+h, x:x+w]
#             enhanced_face = enhance_image(cropped_face)
#             file_path = os.path.join(registered_faces_folder, f"{name}_{roll_no}_{department}_{year}.jpg")
#             cv2.imwrite(file_path, enhanced_face)
#             st.success("✅ Student Registered Successfully!")
#             load_registered_faces()
#         else:
#             st.error("⚠️ Please upload an image with exactly one face.")

# def view_registered_faces():
#     st.header("📂 Registered Students")
#     cols = st.columns(5)
#     for idx, face in enumerate(registered_face_names):
#         with cols[idx % 5]:
#             st.image(
#                 os.path.join(registered_faces_folder, face['file']),
#                 caption=f"{face['name']}\n{face['roll_no']}",
#                 width=100
#             )

# def recognize_live_camera():
#     st.header("🎥 Live Face Recognition")
#     cap = cv2.VideoCapture(0)
#     frame_placeholder = st.empty()
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             st.error("Camera not working!")
#             break
        
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         face_locations = face_recognition.face_locations(rgb_frame)
#         face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
#         for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#             matches = face_recognition.compare_faces(registered_face_encodings, face_encoding, tolerance=0.5)
#             name = "Unknown"
            
#             if True in matches:
#                 match_index = matches.index(True)
#                 name = registered_face_names[match_index]['name']
            
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#             cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
#         frame_placeholder.image(frame, channels="BGR")
    
#     cap.release()
#     cv2.destroyAllWindows()

# def main():
#     with st.sidebar:
#         st.title("📌 Menu")
#         menu = option_menu(
#             menu_title="Choose an Option",
#             options=["Register Student", "Recognize Attendance", "View Registered", "About"],
#             icons=["person-plus", "camera", "list-task", "info-circle"],
#             menu_icon="cast",
#             default_index=0,
#             orientation="vertical"
#         )

#     st.title("🎓 University Attendance System")
#     st.markdown("---")

#     if menu == "Register Student":
#         register_face()
#     elif menu == "Recognize Attendance":
#         recognize_live_camera()
#     elif menu == "View Registered":
#         view_registered_faces()
#     elif menu == "About":
#         st.markdown("""
#         - **University Attendance System**
#         - Uses advanced Face Recognition technology.
#         - Secure, efficient, and easy-to-use.
#         - Developed using **Streamlit** and **OpenCV**.
#         """)

# if __name__ == '__main__':
#     main()


# import streamlit as st
# import cv2
# import face_recognition
# import numpy as np
# from PIL import Image
# import os
# import imutils
# from streamlit_option_menu import option_menu

# st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# registered_faces_folder = "registered_students"
# if not os.path.exists(registered_faces_folder):
#     os.makedirs(registered_faces_folder)

# registered_face_encodings = []
# registered_face_names = []

# def load_registered_faces():
#     global registered_face_encodings, registered_face_names
#     registered_face_encodings.clear()
#     registered_face_names.clear()
    
#     for file_name in os.listdir(registered_faces_folder):
#         if file_name.endswith(".jpg"):
#             parts = file_name.replace(".jpg", "").split("_")
#             if len(parts) == 4:
#                 name, roll_no, department, year = parts
#                 img_path = os.path.join(registered_faces_folder, file_name)
#                 img = face_recognition.load_image_file(img_path)
#                 face_locations = face_recognition.face_locations(img)
#                 if face_locations:
#                     encoding = face_recognition.face_encodings(img, face_locations)[0]
#                     registered_face_encodings.append(encoding)
#                     registered_face_names.append({
#                         'name': name, 'roll_no': roll_no, 'department': department, 'year': year, 'file': file_name
#                     })

# load_registered_faces()

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# def enhance_image(image):
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     image = imutils.resize(image, width=400)
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#     image = cv2.filter2D(image, -1, kernel)
#     return image

# def register_face():
#     st.header("📝 Register Student")
#     name = st.text_input("Student Name")
#     roll_no = st.text_input("Roll Number")
#     department = st.text_input("Department")
#     year = st.selectbox("Year", ["1st", "2nd", "3rd", "4th"])
#     image = st.file_uploader("Upload Image", type=['jpg', 'png'])
    
#     if st.button("Register") and image is not None and name and roll_no and department and year:
#         img = np.array(Image.open(image))
#         faces = face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 1.3, 5)
#         if len(faces) == 1:
#             x, y, w, h = faces[0]
#             cropped_face = img[y:y+h, x:x+w]
#             enhanced_face = enhance_image(cropped_face)
#             file_path = os.path.join(registered_faces_folder, f"{name}_{roll_no}_{department}_{year}.jpg")
#             cv2.imwrite(file_path, enhanced_face)
#             st.success("✅ Student Registered Successfully!")
#             load_registered_faces()
#         else:
#             st.error("⚠️ Please upload an image with exactly one face.")

# def view_registered_faces():
#     st.header("📂 Registered Students")
#     cols = st.columns(5)
#     for idx, face in enumerate(registered_face_names):
#         with cols[idx % 5]:
#             st.image(
#                 os.path.join(registered_faces_folder, face['file']),
#                 caption=f"{face['name']}\n{face['roll_no']}",
#                 width=100
#             )

# def recognize_live_camera():
#     st.header("🎥 Live Face Recognition")
#     cap = cv2.VideoCapture(0)
#     frame_placeholder = st.empty()
#     details_placeholder = st.empty()
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             st.error("Camera not working!")
#             break
        
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         face_locations = face_recognition.face_locations(rgb_frame)
#         face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#         detected_faces = []
        
#         for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#             matches = face_recognition.compare_faces(registered_face_encodings, face_encoding, tolerance=0.5)
#             name = "Unknown"
#             details = "Unknown"
            
#             if True in matches:
#                 match_index = matches.index(True)
#                 matched_person = registered_face_names[match_index]
#                 name = matched_person['name']
#                 details = f"Roll No: {matched_person['roll_no']}\nDepartment: {matched_person['department']}\nYear: {matched_person['year']}"
            
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#             cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#             detected_faces.append(f"**{name}**\n{details}")
        
#         frame_placeholder.image(frame, channels="BGR")
#         details_placeholder.markdown("\n\n".join(detected_faces))
    
#     cap.release()
#     cv2.destroyAllWindows()

# def main():
#     with st.sidebar:
#         st.title("📌 Menu")
#         menu = option_menu(
#             menu_title="Choose an Option",
#             options=["Register Student", "Recognize Attendance", "View Registered", "About"],
#             icons=["person-plus", "camera", "list-task", "info-circle"],
#             menu_icon="cast",
#             default_index=0,
#             orientation="vertical"
#         )

#     st.title("🎓 University Attendance System")
#     st.markdown("---")

#     if menu == "Register Student":
#         register_face()
#     elif menu == "Recognize Attendance":
#         recognize_live_camera()
#     elif menu == "View Registered":
#         view_registered_faces()
#     elif menu == "About":
#         st.markdown("""
#         - **University Attendance System**
#         - Uses advanced Face Recognition technology.
#         - Secure, efficient, and easy-to-use.
#         - Developed using **Streamlit** and **OpenCV**.
#         """)

# if __name__ == '__main__':
#     main()


import streamlit as st
import cv2
import face_recognition
import numpy as np
from PIL import Image
import os
import imutils
from streamlit_option_menu import option_menu
import json
import requests
from streamlit_lottie import st_lottie

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_register = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_jcikwtux.json")
lottie_recognition = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_hq6m4wvk.json")
lottie_view = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_wn9g0zbo.json")
lottie_about = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_uh5akpn3.json")

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
    st.lottie(lottie_register, height=200)
    name = st.text_input("Student Name")
    roll_no = st.text_input("Roll Number")
    department = st.text_input("Department")
    year = st.selectbox("Year", ["1st", "2nd", "3rd", "4th"])
    image = st.file_uploader("Upload Image", type=['jpg', 'png'])
    
    if st.button("Register", key="register") and image is not None and name and roll_no and department and year:
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
    st.lottie(lottie_view, height=200)
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
    st.lottie(lottie_recognition, height=200)
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

    if menu == "Register Student":
        register_face()
    elif menu == "Recognize Attendance":
        recognize_live_camera()
    elif menu == "View Registered":
        view_registered_faces()
    elif menu == "About":
        st.lottie(lottie_about, height=200)
        st.markdown("""
        - **University Attendance System**
        - Uses advanced Face Recognition technology.
        - Secure, efficient, and easy-to-use.
        - Developed using **Streamlit** and **OpenCV**.
        """)

if __name__ == '__main__':
    main()
