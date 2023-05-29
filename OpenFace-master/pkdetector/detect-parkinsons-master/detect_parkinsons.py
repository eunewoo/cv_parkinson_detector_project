#converted code state

# USASGE
# python detect_parkinsons.py --dataset dataset/spiral
# python detect_parkinsons.py --dataset dataset/wave

import numpy as np
import cv2
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import build_montages
from imutils import paths

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

trainingPath = 'dataset/spiral/training/'
testingPath = 'dataset/spiral/testing/'

print("[INFO] loading data...")
(trainX, trainY) = load_split(trainingPath)
(testX, testY) = load_split(testingPath)

le = LabelEncoder()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

print("[INFO] training model...")
model = RandomForestClassifier(n_estimators=100)
model.fit(trainX, trainY)

print("[INFO] evaluating model...")
predictions = model.predict(testX)

cm = confusion_matrix(testY, predictions).flatten()
(tn, fp, fn, tp) = cm
acc = (tp + tn) / float(cm.sum())
sensitivity = tp / float(tp + fn)
specificity = tn / float(tn + fp)

print(f"Accuracy: {acc:.4f}")

testingPaths = list(paths.list_images(testingPath))
idxs = np.random.choice(np.arange(0, len(testingPaths)), size=(25,), replace=False)
images = []

for i in idxs:
    image = cv2.imread(testingPaths[i])
    output = image.copy()
    output = cv2.resize(output, (128, 128))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold(image, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    features = quantify_image(image)
    preds = model.predict([features])
    label = le.inverse_transform(preds)[0]

    color = (0, 255, 0) if label == "healthy" else (0, 0, 255)
    cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        color, 2)
    images.append(output)

montage = build_montages(images, (128, 128), (5, 5))[0]

new_image_path = 'dataset/spiral/testing/parkinson/V01PE01.png'

print("[INFO] loading and preprocessing new image...")

new_image = cv2.imread(new_image_path)
new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
new_image = cv2.resize(new_image, (200, 200))
new_image = cv2.threshold(new_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

new_features = quantify_image(new_image)

print("[INFO] predicting for new image...")
new_preds = model.predict([new_features])
new_label = le.inverse_transform(new_preds)[0]

print("The predicted label for the new image is: ", new_label)



