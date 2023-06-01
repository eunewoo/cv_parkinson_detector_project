import numpy as np
import cv2
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import paths
import joblib  # used for saving and loading sklearn models

def quantify_image(image):
    features = feature.hog(image, orientations=9,
        pixels_per_cell=(10, 10), cells_per_block=(2, 2),
        transform_sqrt=True, block_norm="L1")
    return features

def load_split(path):
    imagePaths = list(paths.list_images(path))
    data = []
    labels = []

    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        image = cv2.threshold(image, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        features = quantify_image(image)
        data.append(features)
        labels.append(label)

    return (np.array(data), np.array(labels))

trainingPath = '/Users/eunewoo/Desktop/2023Spring/CSE327/diary30_327front/OpenFace-master/pkdetector/detect-parkinsons-master/dataset/spiral/training/'
testingPath = '/Users/eunewoo/Desktop/2023Spring/CSE327/diary30_327front/OpenFace-master/pkdetector/detect-parkinsons-master/dataset/spiral/testing/'

print("[INFO] loading data...")
(trainX, trainY) = load_split(trainingPath)
(testX, testY) = load_split(testingPath)

le = LabelEncoder()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

print("[INFO] training model...")
model = RandomForestClassifier(n_estimators=100)
model.fit(trainX, trainY)

# Save the trained model
joblib.dump(model, 'trained_model.pkl')

# Save the label encoder
joblib.dump(le, 'label_encoder.pkl')

# Evaluate the model
print("[INFO] evaluating model...")
predictions = model.predict(testX)
cm = confusion_matrix(testY, predictions).flatten()
(tn, fp, fn, tp) = cm
acc = (tp + tn) / float(cm.sum())
print(f"Accuracy: {acc:.4f}")
