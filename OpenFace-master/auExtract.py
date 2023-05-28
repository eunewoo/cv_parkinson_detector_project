import csv
import statistics

# Define the AUs of interest
aus = [1, 2, 4]

# Load the OpenFace output file into a list of dictionaries
with open('processed/surpriseMe.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

# For each AU, calculate the variance of the raw AU value when the AU is active
for au in aus:
    au_r = f"AU{au:02}_r"
    au_c = f"AU{au:02}_c"

    # Select only the frames where the AU is active
    active_frames = [float(row[au_r]) for row in data if float(row[au_c]) == 1]

    # Calculate the variance of the raw AU value for these frames, if there are any
    if active_frames:
        variance = statistics.variance(active_frames)
        print(f"The variance of {au_r} when active is {variance}")


