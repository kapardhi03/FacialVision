# FacialVision

## Face Detection and Recognition with OpenCV

This repository demonstrates the implementation of facial detection and recognition using OpenCV. The project is divided into two parts:
1. **Face Detection**: Detecting faces in images using Haar Cascades.
2. **Face Recognition**: Recognizing faces using Local Binary Patterns Histograms (LBPH) with OpenCV.

## Prerequisites

- Python 3.x
- OpenCV (with Contrib modules)
- NumPy

You can install the necessary packages using:
```bash
pip install opencv-contrib-python numpy
```

## Directory Structure

```
face_recognition/
│
├── face_recog.py             # Script for face detection and recognition
├── faces_train.py            # Script for training the face recognizer
├── haar_face.xml             # Haar Cascade for face detection
├── features.npy              # Numpy array of face features (generated by training script)
├── labels.npy                # Numpy array of labels (generated by training script)
├── face_trained.yml          # Trained model file (generated by training script)
└── images/
    ├── person1/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── person2/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    └── ...
```

## Explanation

### Face Detection and Recognition

The `face_recog.py` script performs face detection and recognition on a given image. It uses a pre-trained LBPH face recognizer model to identify the person in the image.

### Training the Face Recognizer

The `faces_train.py` script is used to train the face recognizer. Here’s what it does:
1. **Load Images**: It iterates through the images stored in the `images` directory. Each subdirectory within `images` should be named after the person and contain images of that person.
2. **Face Detection**: For each image, faces are detected using the Haar Cascade classifier (`haar_face.xml`).
3. **Extract Features**: The regions of interest (faces) are extracted and converted to grayscale.
4. **Create Training Data**: The grayscale face images (features) and corresponding labels (person's name) are stored in lists.
5. **Convert to NumPy Arrays**: These lists are then converted to NumPy arrays and saved as `features.npy` and `labels.npy` for later use.
6. **Train the Recognizer**: An LBPH face recognizer is trained using the features and labels.
7. **Save the Model**: The trained model is saved as `face_trained.yml`.

### .npy Files

- **features.npy**: This file contains a NumPy array of the face regions (features) detected from the training images. It is used to store the training data for the face recognizer.
- **labels.npy**: This file contains a NumPy array of labels corresponding to each face region in `features.npy`. These labels are indices that map to the names of the persons (directories).

### NOTE

While OpenCV's built-in face recognizer (LBPH) is convenient and easy to use, it may not provide the highest accuracy for face recognition tasks, especially with larger and more diverse datasets. For more robust and accurate face recognition, consider using deep learning-based methods, such as those provided by libraries like Dlib or face_recognition, which leverage pre-trained models.

## Running the Project

### Training the Model

1. Place your training images in the `images` directory, organized by subdirectory for each person.
2. Run the training script:
```bash
python faces_train.py
```

This will generate the `features.npy`, `labels.npy`, and `face_trained.yml` files.

### Detecting and Recognizing Faces

1. Ensure the `face_recog.py` script points to an image file you want to test.
2. Run the face recognition script:
```bash
python face_recog.py
```

This will perform face detection and recognition on the specified image and display the results.
