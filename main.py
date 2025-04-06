import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Conv2D, Dense, MaxPooling2D, BatchNormalization
from keras.models import Sequential
from imblearn.over_sampling import SMOTE, ADASYN


df = pd.read_csv("C:/Users/Shaun/emotions project test/test/train.csv")
 


print(df.head())


x_train = df['pixels']
y_train = df['emotion']

for i, row in enumerate(x_train):
    length = len(row.split())
    print(f"Row {i} has {length} pixels")


EXPECTED_PIXEL_COUNT = 48 * 48  # 2304

filtered_data = [
    (pixels, emotion)
    for pixels, emotion in zip(df['pixels'], df['emotion'])
    if len(pixels.split()) == EXPECTED_PIXEL_COUNT
]

# Separate pixels and labels
x_train = [list(map(int, pixels.split())) for pixels, _ in filtered_data]
y_train = [emotion for _, emotion in filtered_data]

# Convert to numpy arrays
x_train_array = np.array(x_train, dtype=np.uint8)
y_train = np.array(y_train, dtype=np.uint8)


ad = ADASYN(random_state=2)
x_train_bal, y_train_bal = ad.fit_resample(x_train_array, y_train)

x_train_image = np.array(x_train_bal).reshape(-1, 48, 48)

print("Image shape:", x_train_image.shape)
plt.imshow(x_train_image[0], cmap='gray')
plt.title(f"Label: {y_train_bal[0]}")
plt.show()

