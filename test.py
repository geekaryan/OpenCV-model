import cv2
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Directory to save captured face images
data_dir = './images'

# Create the directory if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Initialize the OpenCV face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the face data list
face_data = []

# Initialize the label list
labels = []

# Initialize the label encoder
label_encoder = LabelEncoder()

# Capture face images
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around the detected faces and capture the face data
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the face region from the frame
        face_roi = gray[y:y + h, x:x + w]

        # Resize the face region to a fixed size (e.g., 160x160)
        face_roi = cv2.resize(face_roi, (160, 160))

        # Add the face data to the list
        face_data.append(face_roi)

        # Add the corresponding label (you can prompt the user for the label)
        labels.append('person')

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Wait for the 'c' key to be pressed to capture an image
    if cv2.waitKey(1) & 0xFF == ord('c'):
        print('Captured image')

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
video_capture.release()
cv2.destroyAllWindows()

# Convert the face data and labels to NumPy arrays
face_data = np.array(face_data)
labels = np.array(labels)

# Preprocess the labels
labels = label_encoder.fit_transform(labels)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(face_data, labels, test_size=0.2, random_state=42)

# Define the FaceNet model architecture
def create_facenet_model(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(len(label_encoder.classes_), activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Create and compile the FaceNet model
input_shape = X_train[0].shape
facenet_model = create_facenet_model(input_shape)
facenet_model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the FaceNet model
checkpoint = ModelCheckpoint('facenet_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
facenet_model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=10, callbacks=[checkpoint])

# Evaluate the trained model
best_model = create_facenet_model(input_shape)
best_model.load_weights('facenet_model.h5')

y_pred = best_model.predict(X_val)
y_pred = np.argmax(y_pred, axis=1)
val_accuracy = accuracy_score(y_val, y_pred)
print('Validation Accuracy:', val_accuracy)
