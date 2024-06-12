import os
import numpy as np
import cv2 as cv

# Load the Haar cascade classifier
haar_cascade = cv.CascadeClassifier('../haar_face.xml')

# Load people's names from the images directory
people = []
for person in os.listdir("/Users/kapardhikannekanti/images"):
    people.append(person)

# Remove any unwanted files like '.DS_Store' if present
if '.DS_Store' in people:
    people.remove('.DS_Store')

# Load pre-trained features and labels
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)

# Create an LBPH face recognizer and read the trained model
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img_path = '/Users/kapardhikannekanti/images/Kapardhi/K.Kapardhi.jpeg'
img = cv.imread(img_path)

# Check if the image was successfully loaded
if img is None:
    print(f"Error: Unable to read image at {img_path}")
else:
    # Convert the image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("Gray", gray)

    # Detect faces in the image
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+w]
        
        # Predict the label and confidence of the detected face
        label, confidence = face_recognizer.predict(faces_roi)
        print(f"Label = {people[label]} with confidence {confidence}")

        # Annotate the image with the predicted label and draw a rectangle around the face
        cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, color=(0, 255, 0), thickness=2)
        cv.rectangle(img, pt1=(x, y), pt2=(x+w, y+h), color=(0, 255, 0), thickness=2)

    # Display the annotated image
    cv.imshow("Detected Face", img)

    # Wait for a key press to close the windows
    cv.waitKey(12000)
    cv.destroyAllWindows()
