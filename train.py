import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from model_builder import build_model
import json
import os

dataset_path = "dataset"  
img_size = (96, 96)       
batch_size = 16
epochs = 30
train_split = 0.70
val_split = 0.15
test_split = 0.15

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f" Using GPU: {physical_devices}")
else:
    print("⚠ No GPU detected, training will run on CPU.")

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=(val_split + test_split),
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=(test_split / (val_split + test_split))
)

val_generator = val_test_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

label_map = train_generator.class_indices
with open("label_mapv2.json", "w") as f:
    json.dump(label_map, f)

model = build_model(input_shape=(img_size[0], img_size[1], 3), num_classes=len(label_map))
model.summary()

checkpoint = ModelCheckpoint(
    "best_modelv2.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    verbose=1
)

def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.9

lr_callback = LearningRateScheduler(lr_schedule)

callbacks = [checkpoint, early_stop, reduce_lr, lr_callback]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks
)

print("\n📊 Evaluating on Test Set...")
loss, acc = model.evaluate(test_generator)
print(f"✅ Test Accuracy: {acc:.4f}")
print(f"✅ Test Loss: {loss:.4f}")

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes
target_names = list(label_map.keys())

print("\n📑 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=target_names))

cm = confusion_matrix(y_true, y_pred_classes)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(25, 25))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues',
            xticklabels=target_names, yticklabels=target_names,
            cbar_kws={'label': 'Proportion'})
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix')
plt.tight_layout()
plt.show()
