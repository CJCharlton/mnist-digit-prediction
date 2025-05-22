import torch
import pandas as pd
import numpy as np
from train import DigitCNN
import os

# Load test data
test_df = pd.read_csv('../data/test.csv')
test_images = test_df.values.astype('float32') / 255.0
X_test = torch.tensor(test_images).reshape(-1, 1, 28, 28)

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DigitCNN().to(device)
model.load_state_dict(torch.load('../models/digit_cnn.pth', map_location=device))
model.eval()

# Predict
with torch.no_grad():
    preds = []
    for i in range(0, len(X_test), 64):
        batch = X_test[i:i+64].to(device)
        outputs = model(batch)
        preds.extend(outputs.argmax(1).cpu().numpy())

# Prepare submission
submission = pd.DataFrame({
    "ImageId": np.arange(1, len(preds) + 1),
    "Label": preds
})

submission.to_csv('../data/submission.csv', index=False)
print("Submission file saved to ../data/submission.csv")