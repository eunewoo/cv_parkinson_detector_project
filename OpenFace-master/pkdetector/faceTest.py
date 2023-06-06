# predict.py
import joblib
import csv
import statistics
import subprocess
import pandas as pd
import json

# Load the saved model
clf = joblib.load('/Users/eunewoo/Desktop/2023Spring/CSE327/diary30_327front/OpenFace-master/pkdetector/model.pkl')

# Load the saved scaler
scaler = joblib.load('/Users/eunewoo/Desktop/2023Spring/CSE327/diary30_327front/OpenFace-master/pkdetector/scaler.pkl')

# Ask for video file input
video_files = ["/Users/eunewoo/Desktop/2023Spring/CSE327/diary30_327front/OpenFace-master/pkdetector/sampleFront/uploads/smileMe.webm", "/Users/eunewoo/Desktop/2023Spring/CSE327/diary30_327front/OpenFace-master/pkdetector/sampleFront/uploads/disgustMe.webm", "/Users/eunewoo/Desktop/2023Spring/CSE327/diary30_327front/OpenFace-master/pkdetector/sampleFront/uploads/surpriseMe.webm"]

# Define the AUs of interest
aus_smile = [1, 6, 12]
aus_disgusted = [4, 7, 9]
aus_surprised = [1, 2, 4]
aus = [aus_smile, aus_disgusted, aus_surprised]
output_files = ['smile.csv', 'disgusted.csv', 'surprised.csv']

# Run the FeatureExtraction command for each video
variances = []
for i in range(3):
    command = f"/Users/eunewoo/Desktop/2023Spring/CSE327/diary30_327front/OpenFace-master/build/bin/FeatureExtraction -f {video_files[i]} -of {output_files[i]}"
    subprocess.run(command, shell=True)

    # Load the OpenFace output file into a list of dictionaries and calculate variance
    with open(f'/Users/eunewoo/Desktop/2023Spring/CSE327/diary30_327front/OpenFace-master/pkdetector/sampleFront/processed/{output_files[i]}', 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    for au in aus[i]:
        au_r = f"AU{au:02}_r"
        au_c = f"AU{au:02}_c"

        # Select only the frames where the AU is active
        active_frames = [float(row[au_r]) for row in data if float(row[au_c]) == 1]

        # Calculate the variance of the raw AU value for these frames, if there are any
        if len(active_frames) > 1:
            variance = statistics.variance(active_frames)
            variances.append(variance)
        else:
            variances.append(0)

# Convert the list to a DataFrame
variances2 = pd.DataFrame([variances], columns=['AU_01_t12', 'AU_06_t12', 'AU_12_t12', 'AU_04_t13', 'AU_07_t13', 'AU_09_t13', 'AU_01_t14', 'AU_02_t14', 'AU_04_t14'])

# Use the loaded scaler to transform the input data
user_feats_scaled = scaler.transform(variances2)

# Make a prediction
print("Scaled input features:", user_feats_scaled)
prediction = clf.predict(user_feats_scaled)
print("Prediction:", prediction)


# Output the result
if prediction[0] == 1:
    result = {"prediction": 1, "message": "Parkinson"}
else:
    result = {"prediction": 0, "message": "Not Parkinson"}

# Write the result to output.json
with open('output.json', 'w') as f:
    json.dump(result, f)
