# predict.py
import joblib
from sklearn.preprocessing import StandardScaler
import csv
import statistics
import subprocess

# Load the saved model
clf = joblib.load('model.pkl')

# Ask for video file input
video_files = []

for _ in range(3):
    filename = input('Please enter 3 facial expression video in order (smile, disgusted, surprised) : ')
    video_files.append(filename)

# Define the AUs of interest
aus_smile = [1, 6, 12]
aus_disgusted = [4, 7, 9]
aus_surprised = [1, 2, 4]
aus = [aus_smile, aus_disgusted, aus_surprised]
output_files = ['smile.csv', 'disgusted.csv', 'surprised.csv']

# Run the FeatureExtraction command for each video
variances = []
for i in range(3):
    command = f"../build/bin/FeatureExtraction -f {video_files[i]} -of {output_files[i]}"
    subprocess.run(command, shell=True)
    
    # Load the OpenFace output file into a list of dictionaries and calculate variance
    with open(f'processed/{output_files[i]}', 'r') as f:
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
            # print(f"The variance of {au_r} when active is {variance}")
        else:
            variances.append(0)
            # print(f"The variance of {au_r} when active is 0")

# Make sure the input data is scaled in the same way as the training data was
user_feats_scaled = StandardScaler().fit_transform([variances])

# Make a prediction
prediction = clf.predict(user_feats_scaled)

# Output the result
if prediction[0] == 1:
    print("Trained model predicts that you have Parkinson's disease.")
else:
    print("Trained model predicts that you do not have Parkinson's disease.")
