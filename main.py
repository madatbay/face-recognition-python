import cv2

# Load pretrained face frontal xml file
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Read video data ==> if prop is 0 - this 'll use default camera and you can pass video path, too
cam = cv2.VideoCapture(0)

# Start infinite while loop to read each frame
while True:
    # Read each frame during the loop
    successful_frame, frame = cam.read()

    # Convert frame to Gray Scale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face in the frame
    face_coords = trained_face_data.detectMultiScale(gray_frame)

    # Draw rectangle on the frame
    for (x, y, w, h) in face_coords:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show Frame
    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    # Break the loop on key press; q or Q
    if key == 81 or key == 113:
        break

# Release the video capture
cam.release()
