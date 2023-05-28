import csv
import statistics
import subprocess
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Ask for video file input
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
video_files = []
for _ in range(3):
    filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    video_files.append(filename)

# Define the AUs of interest
aus_smile = [1, 6, 12]
aus_disgusted = [4, 7, 9]
aus_surprised = [1, 2, 4]

aus = [aus_smile, aus_disgusted, aus_surprised]
output_files = ['smile.csv', 'disgusted.csv', 'surprised.csv']

# Run the FeatureExtraction command for each video
for i in range(3):
    command = f"../build/bin/FeatureExtraction -f {video_files[i]} -of {output_files[i]}"
    subprocess.run(command, shell=True)

# Load the OpenFace output file into a list of dictionaries and calculate variance
for i in range(3):
    with open(f'processed/{output_files[i]}', 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    for au in aus[i]:
        au_r = f"AU{au:02}_r"
        au_c = f"AU{au:02}_c"

        # Select only the frames where the AU is active
        active_frames = [float(row[au_r]) for row in data if float(row[au_c]) == 1]

        # Calculate the variance of the raw AU value for these frames, if there are any
        if active_frames:
            variance = statistics.variance(active_frames)
            print(f"The variance of {au_r} when active in {output_files[i]} is {variance}")
