from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

def load_data(dataset_path='dataset', img_size=(64, 64), batch_size=32):
    seed = 42
    train_split = 0.70
    val_split = 0.15
    test_split = 0.15

    datagen = ImageDataGenerator(rescale=1./255, validation_split=(1 - train_split))

    train_data = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=seed
    )

  
    temp_data = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=1,  
        class_mode='categorical',
        subset='validation',
        shuffle=True,
        seed=seed
    )

    total_temp = temp_data.samples
    val_size = int(total_temp * (val_split / (val_split + test_split)))

    x_temp = []
    y_temp = []
    for _ in range(total_temp):
        img, label = temp_data.next()
        x_temp.append(img[0])
        y_temp.append(label[0])
    x_temp = np.array(x_temp)
    y_temp = np.array(y_temp)

    x_val, x_test = x_temp[:val_size], x_temp[val_size:]
    y_val, y_test = y_temp[:val_size], y_temp[val_size:]

    return train_data, (x_val, y_val), (x_test, y_test)
