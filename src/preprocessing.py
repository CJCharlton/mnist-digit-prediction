import pandas as pd
import numpy as np

train_df = pd.read_csv('../data/train.csv')

images = train_df.drop('label', axis=1).values
images = images.reshape(-1, 28, 28)

digits = train_df['label'].values