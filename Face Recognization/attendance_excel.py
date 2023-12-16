import cv2
import face_recognition
import numpy as np
import os
import datetime
import PySimpleGUI as sg

# Path to the database folder
database_path = 'D:\Face Recognization\database'  # Replace with the path to your database folder

# Path to the attendance file
attendance_path = 'D:/Face Recognization/attendance.txt'  # Replace with the path to your attendance file

# Load sample pictures and learn how to recognize them.
known_face_encodings = []
known_face_names = []

# Iterate over the images in the database folder
for filename in os.listdir(database_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Add other file types if needed
        # Load the image file and get face encodings
        image_path = os.path.join(database_path, filename)
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0]
        
        # Add face encoding and name (from the file name) to the lists
        known_face_encodings.append(face_encoding)
        known_face_names.append(filename.split('.')[0])  # Assumes the file name is the person's name

# Get the current date
today = datetime.date.today()

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Initialize a variable to store the previous message and time
prev_message = {}
prev_time = {}

# Initialize a variable to store the time interval for showing message and attendance
time_interval = 60 * 60 # One hour in seconds

# Create a layout for the window
layout = [
    [sg.Text("Face Recognition Attendance System", font=("Arial", 18))],
    [sg.Image(filename="", key="image")], # A blank image to display the webcam feed
    [sg.Text("Name: ", size=(15, 1)), sg.Text("", key="name", size=(15, 1))], # A text widget to display the name
    [sg.Text("Attendance: ", size=(15, 1)), sg.Text("", key="attendance", size=(15, 1))], # A text widget to display the attendance status
    [sg.Button("Exit", size=(10, 1))]
]

# Create a window with the layout
window = sg.Window("Face Recognition", layout)

# Load a video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Read the window events and values
    event, values = window.read(timeout=20)

    # Check if the user wants to exit
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Update the attendance file
        if name != "Unknown":
            # Get the current time
            now = datetime.datetime.now()

            # Check if the name is in the previous message dictionary
            if name in prev_message:
                # Get the previous message and time
                message = prev_message[name]
                prev = prev_time[name]

                # Check if the message is "Attendance marked"
                if message == "Attendance marked":
                    # Check if the time difference is less than the time interval
                    if (now - prev).total_seconds() < time_interval:
                        # Do nothing
                        pass
                    else:
                        # Update the previous message and time
                        prev_message[name] = "Attendance updated"
                        prev_time[name] = now

                        # Show a message box to the user
                        sg.Popup(f"{name}, your attendance is updated at {now.strftime('%H:%M')}")

                        # Write the name, date, and status to the file, separated by commas
                        with open(attendance_path, 'a') as f:
                            f.write(f'{name},{today},Present\n')
                else:
                    # Update the previous message and time
                    prev_message[name] = "Attendance marked"
                    prev_time[name] = now

                    # Show a message box to the user
                    sg.Popup(f"{name}, your attendance is marked at {now.strftime('%H:%M')}")

                    # Write the name, date, and status to the file, separated by commas
                    with open(attendance_path, 'a') as f:
                        f.write(f'{name},{today},Present\n')
            else:
                # Add the name, message, and time to the previous message and time dictionaries
                prev_message[name] = "Attendance marked"
                prev_time[name] = now

                # Show a message box to the user
                sg.Popup(f"{name}, your attendance is marked at {now.strftime('%H:%M')}")

                # Write the name, date, and status to the file, separated by commas
                with open(attendance_path, 'a') as f:
                    f.write(f'{name},{today},Present\n')

            # Update the name and attendance widgets on the window
            window["name"].update(name)
            window["attendance"].update("Present")
        else:
            # Update the name and attendance widgets on the window
            window["name"].update("Unknown")
            window["attendance"].update("Absent")

    # Convert the frame to PNG format
    imgbytes = cv2.imencode(".png", frame)[1].tobytes()

    # Update the image widget on the window with the frame
    window["image"].update(data=imgbytes)

# Release handle to the webcam
video_capture.release()

# Close the window
window.close()
