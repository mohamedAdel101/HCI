import cv2
import face_recognition
import sqlite3

# Load Haar cascade for face detection
Haarcascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Paths to known images
known_image_paths = [
    "C:/Users/moham/Downloads/Mabrouk.jpg", 
    "C:/Users/moham/Downloads/Badr.jpg",
    "C:/Users/moham/Downloads/Ahmed.jpg",
    "C:/Users/moham/Downloads/Mohamed.jpg"
]

# Load and encode all known faces
known_face_encodings = []
known_face_names = []

for image_path in known_image_paths:
    known_image = face_recognition.load_image_file(image_path)
    known_face_encoding = face_recognition.face_encodings(known_image)[0]
    known_face_encodings.append(known_face_encoding)
    known_face_names.append(image_path.split("/")[-1].split(".")[0])  # Extract name from image path

# Open the video capture (use 0 for webcam or a file path for video file)
cap = cv2.VideoCapture(0)

# Check if video capture is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    last_known_name = None  # Store the last known face's name for task display
    tasks_result = []  # Store tasks for the current face
    
    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Convert the frame to grayscale for face detection
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image using Haar cascade
        faces = Haarcascade.detectMultiScale(
            img_gray,
            scaleFactor=1.1,   # Adjust the scale factor
            minNeighbors=5,    # Minimum number of neighbors for a valid face
            minSize=(30, 30)   # Minimum size of faces to detect
        )

        # Check if faces were detected
        if len(faces) == 0:
            print("No faces detected.")
            last_known_name = None  # Clear last known name when no face is detected
            tasks_result = []  # Clear tasks when no face is detected
        else:
            print(f"Detected {len(faces)} faces.")
            for (x, y, w, h) in faces:
                # Extract the face from the frame
                face_image = frame[y:y + h, x:x + w]
                
                # Convert the face image to RGB (face_recognition requires RGB)
                rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                
                # Get face encodings for the detected face
                face_encodings = face_recognition.face_encodings(rgb_face_image)
                
                if face_encodings:
                    face_encoding = face_encodings[0]
                    
                    # Compare the detected face with all known faces
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    
                    known_name = "Unknown"
                    tasks_result = []  # Reset tasks result for each frame
                    if True in matches:
                        # If a match is found, get the index of the matched face
                        first_match_index = matches.index(True)
                        known_name = known_face_names[first_match_index]  # Use the name of the matched face

                        # Query the database to get tasks for the employee
                        try:
                            sqliteConnection = sqlite3.connect("C:/Users/moham/Downloads/TeamMGT.db")
                            cursor = sqliteConnection.cursor()
                            cursor2 = sqliteConnection.cursor()

                            # Query to get the employee ID by name
                            query = "SELECT id FROM Employess WHERE name = ?;"
                            cursor.execute(query, (known_name,))
                            result = cursor.fetchall()

                            if result:  # If result is not empty, proceed
                                print('DB Loaded')
                                print(result)
                                
                                # Query to get tasks assigned to the employee
                                select_task_query = "SELECT t_name, t_sDate, t_eDate FROM Tasks WHERE E_id = ?;"
                                cursor2.execute(select_task_query, (result[0][0],))  # Pass employee ID to the query
                                tasks_result = cursor2.fetchall()
                                print(tasks_result)
                            else:
                                tasks_result = []
                        except sqlite3.Error as error:
                            print("Error while interacting with SQLite:", error)
                        finally:
                            # Always close the connection
                            if sqliteConnection:
                                sqliteConnection.close()
                    
                    # Draw a rectangle around the face in the original frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, known_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    # Calculate dynamic table width based on task lengths
                    max_task_len = max([len(task[0]) for task in tasks_result], default=0)
                    max_start_len = max([len(task[1]) for task in tasks_result], default=0)
                    max_end_len = max([len(task[2]) for task in tasks_result], default=0)
                    
                    # Define minimum and maximum column widths
                    task_width = max(max_task_len * 15, 100)
                    date_width = max(max(max_start_len, max_end_len) * 15, 100)

                    # Calculate the table width
                    table_width = task_width + date_width * 2 + 40  # Padding and column spacing
                    table_height = 60 + len(tasks_result) * 40  # Table height based on rows
                    
                    # Calculate the required window size
                    window_width = frame.shape[1]
                    window_height = frame.shape[0] + table_height + 20  # Add space for the table at the bottom
                    
                    # Resize the window to fit the frame and table
                    cv2.resizeWindow("Face Detection", window_width, window_height)
                    
                    # Define table position
                    table_x = frame.shape[1] // 2 - table_width // 2
                    table_y = frame.shape[0] - table_height - 10
                    
                    # Add background rectangle for the table
                    cv2.rectangle(frame, (table_x, table_y), (table_x + table_width, table_y + table_height), (255, 255, 255), -1)
                    
                    # Draw table headers
                    header_font = cv2.FONT_HERSHEY_SIMPLEX
                    header_font_size = 0.8
                    cv2.putText(frame, "Task", (table_x + 20, table_y + 30), header_font, header_font_size, (0, 0, 0), 2)
                    cv2.putText(frame, "Start Date", (table_x + task_width, table_y + 30), header_font, header_font_size, (0, 0, 0), 2)
                    cv2.putText(frame, "End Date", (table_x + task_width + date_width, table_y + 30), header_font, header_font_size, (0, 0, 0), 2)

                    # Draw task rows
                    row_y_offset = table_y + 50
                    row_font_size = 0.7
                    for task in tasks_result:
                        cv2.putText(frame, task[0], (table_x + 20, row_y_offset), header_font, row_font_size, (0, 0, 0), 2)
                        cv2.putText(frame, task[1], (table_x + task_width, row_y_offset), header_font, row_font_size, (0, 0, 0), 2)
                        cv2.putText(frame, task[2], (table_x + task_width + date_width, row_y_offset), header_font, row_font_size, (0, 0, 0), 2)
                        row_y_offset += 40  # Adjust vertical spacing for rows

        # Display the frame with detected faces and table
        cv2.imshow("Face Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close the display window
cap.release()
cv2.destroyAllWindows()
