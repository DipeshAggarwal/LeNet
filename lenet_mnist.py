from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from lenet import LeNet
import matplotlib.pyplot as plt
import numpy as np

print("[INFO] Accessing MNIST...")
(train_data, train_label), (testing_data, testing_label) = mnist.load_data()

if K.image_data_format() == "channels_first":
    train_data = train_data.reshape(train_data.shape[0], 1, 28, 28)
    testing_data = testing_data.reshape(testing_data.shape[0], 1, 28, 28)
else:
    train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
    testing_data = testing_data.reshape(testing_data.shape[0], 28, 28, 1)
    
train_data = train_data.astype("float32") / 255.0
testing_data = testing_data.astype("float32") / 255.0

le = LabelBinarizer()
train_label = le.fit_transform(train_label)
testing_label = le.transform(testing_label)

print("[INFO] Compiling Model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] Training Network...")
H = model.fit(train_data, train_label, validation_data=(testing_data, testing_label), batch_size=128, epochs=20, verbose=1)

print("[INFO] Evaluating Network...")
predictions = model.predict(testing_data, batch_size=128)
print(classification_report(testing_label.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in le.classes_]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="Train Loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="Validation Loss")
plt.plot(np.arange(0, 20), H.history["accuracy"], label="Training Accuracy")
plt.plot(np.arange(0, 20), H.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
