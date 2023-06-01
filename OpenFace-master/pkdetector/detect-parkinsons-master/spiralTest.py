import numpy as np
import cv2
from skimage import feature
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imutils import paths
import joblib  # used for loading sklearn models

def quantify_image(image):
    features = feature.hog(image, orientations=9,
        pixels_per_cell=(10, 10), cells_per_block=(2, 2),
        transform_sqrt=True, block_norm="L1")
    return features

# Load the trained model and label encoder
model = joblib.load('trained_model.pkl')
le = joblib.load('label_encoder.pkl')

# Load the new image and make the prediction
new_image_path = input("Please enter the path of the new image: ")
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
