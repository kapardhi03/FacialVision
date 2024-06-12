import os
import cv2 as cv
import numpy as np

people = []
for image in os.listdir("/Users/kapardhikannekanti/images"):
    people.append(image)

people.remove('.DS_Store')
print("People:", people)
DIR = r'/Users/kapardhikannekanti/images'


haar_cascade = cv.CascadeClassifier('../haar_face.xml')



features = []
labels = []

def create_train():
    for image in people:
        Path = os.path.join(DIR, image)
        label = people.index(image)

        for img in os.listdir(Path):
            img_path = os.path.join(Path, img)

            img_array = cv.imread(img_path)
            gray =cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for(x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

# /Users/kapardhikannekanti/images/Abhinav.png
create_train()

# print(f"Length of features = {len(features)}")
# print(f"Length of labels = {len(labels)}")

print("-*-"*10,"Training Done","-*-"*10)
## convert to numpy array
features = np.array(features, dtype='object')
labels = np.array(labels)

# Create an LBPH face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the recognizer on the features (face images) and labels
face_recognizer.train(features, labels)

# Save the trained model
face_recognizer.save('face_trained.yml')

# Save the features and labels arrays for future use
np.save("features.npy", features)
np.save("labels.npy", labels)