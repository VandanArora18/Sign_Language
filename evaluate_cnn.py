import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt

dataset_path = "dataset"
img_size = (96, 96)
batch_size = 16
val_split = 0.15
model_path = "best_modelv2.h5"
label_map_path = "label_mapv2.json"

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"✅ Using GPU: {physical_devices}")
else:
    print("⚠ No GPU detected, evaluation will run on CPU.")

print("\nLoading model...")
model = load_model(model_path)

with open(label_map_path, "r") as f:
    label_map = json.load(f)
target_names = list(label_map.keys())

datagen = ImageDataGenerator(rescale=1./255, validation_split=val_split)

test_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print("\nEvaluating on test set...")
loss, acc = model.evaluate(test_generator)
print(f"✅ Test Accuracy: {acc:.4f}")
print(f"✅ Test Loss: {loss:.4f}")

y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

print("\n📑 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=target_names))

cm = confusion_matrix(y_true, y_pred_classes)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(25, 25))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names,
            cbar_kws={'label': 'Proportion'})
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix')
plt.tight_layout()
plt.show()