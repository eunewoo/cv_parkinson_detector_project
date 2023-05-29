# Detecting Parkinson’s Disease 

![alt text](https://github.com/rohitdubey03/detect-parkinsons/blob/master/images/Python-machine-learning-project-.png)

## Parkinson’s disease is a nervous system disorder that affects movement. The disease is progressive and is marked by five different stages (source).

Stage 1:Mild symptoms that do not typically interfere with daily life, including tremors and movement issues on only one side of the body.

Stage 2: Symptoms continue to become worse with both tremors and rigidity now affecting both sides of the body. Daily tasks become challenging.

Stage 3: Loss of balance and movements with falls becoming frequent and common. The patient is still capable of (typically) living independently.

Stage 4: Symptoms become severe and constraining. The patient is unable to live alone and requires help to perform daily activities.

Stage 5: Likely impossible to walk or stand. The patient is most likely wheelchair bound and may even experience hallucinations.

## Drawing spirals and waves to detect Parkinson’s disease

A 2017 study by Zham et al. found that it was possible to detect Parkinson’s by asking the patient to draw a spiral and then track:

Speed of drawing

Pen pressure

The researchers found that the drawing speed was slower and the pen pressure lower among Parkinson’s patients — this was especially pronounced for patients with a more acute/advanced forms of the disease.

We’ll be leveraging the fact that two of the most common Parkinson’s symptoms include tremors and muscle rigidity which directly impact the visual appearance of a hand drawn spiral and wave.

The variation in visual appearance will enable us to train a computer vision + machine learning algorithm to automatically detect Parkinson’s disease.


#### After extracting features from the input images we trained a Random Forest classifier with 100 total decision trees in the forest, obtaining:

83.33% accuracy for spiral

71.33% accuracy for the wave

