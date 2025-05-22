import pandas as pd
import numpy as np

# Read the train.csv file from the data directory
train_df = pd.read_csv('../data/train.csv')

# If the first column is 'label', exclude it from the image data
if 'label' in train_df.columns:
    images = train_df.drop('label', axis=1).values
else:
    images = train_df.values

# Reshape to (num_samples, 28, 28)
images = images.reshape(-1, 28, 28)

# images is now a numpy.ndarray of shape (num_samples, 28, 28)