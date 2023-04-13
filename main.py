# Importing the required Keras modules containing model and layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Dropout,
    Flatten,
    MaxPooling2D,
    AveragePooling2D,
    BatchNormalization,
    MaxPool2D,
)
from tensorflow.image import resize
from tensorflow.nn import relu, softmax
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


def reshapeAndConvert(x_train, x_test, size):
    # this reshaping and input shape creation is convenient due to the three models requiring different inputs
    x_train = x_train.reshape(x_train.shape[0], size, size, 1).astype("float32")
    x_test = x_test.reshape(x_test.shape[0], size, size, 1).astype("float32")
    input_shape = (size, size, 1)

    # Normalizing the RGB codes by dividing it into the max RGB value
    x_train /= 255
    x_test /= 255

    return x_train, x_test, input_shape


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test, input_shape = reshapeAndConvert(x_train, x_test, 28)

MLP = Sequential()
MLP.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
MLP.add(Flatten())
MLP.add(Dense(10, activation=softmax))

LeNet5 = Sequential()
LeNet5.add(
    Conv2D(filters=6, kernel_size=(3, 3), activation="relu", input_shape=input_shape)
)
LeNet5.add(AveragePooling2D())

LeNet5.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu"))
LeNet5.add(Flatten())

LeNet5.add(Dense(units=120, activation="relu"))
LeNet5.add(Dense(units=84, activation="relu"))
LeNet5.add(Dense(units=10, activation="softmax"))

# # AlexNet network Layers
# AlexNet = Sequential()
# AlexNet.add(
#     Conv2D(
#         filters=96,
#         kernel_size=(11, 11),
#         strides=(4, 4),
#         activation="relu",
#         input_shape=input_shape,
#     )
# )
# AlexNet.add(BatchNormalization())
# AlexNet.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
# AlexNet.add(
#     Conv2D(
#         filters=256,
#         kernel_size=(5, 5),
#         strides=(1, 1),
#         activation="relu",
#         padding="same",
#     )
# )
# AlexNet.add(BatchNormalization())
# AlexNet.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
# AlexNet.add(
#     Conv2D(
#         filters=384,
#         kernel_size=(3, 3),
#         strides=(1, 1),
#         activation="relu",
#         padding="same",
#     )
# )
# AlexNet.add(BatchNormalization())
# AlexNet.add(
#     Conv2D(
#         filters=384,
#         kernel_size=(3, 3),
#         strides=(1, 1),
#         activation="relu",
#         padding="same",
#     )
# )
# AlexNet.add(BatchNormalization())
# AlexNet.add(
#     Conv2D(
#         filters=384,
#         kernel_size=(3, 3),
#         strides=(1, 1),
#         activation="relu",
#         padding="same",
#     )
# )
# AlexNet.add(BatchNormalization())
# AlexNet.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
# AlexNet.add(Flatten())
# AlexNet.add(Dense(4096, activation="relu"))
# AlexNet.add(Dropout(0.5))
# AlexNet.add(Dense(4096, activation="relu"))
# AlexNet.add(Dropout(0.5))
# AlexNet.add(Dense(10, activation="softmax"))

MLP.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy", "sparse_top_k_categorical_accuracy"],
)
MLP.fit(x=x_train, y=y_train, epochs=1)
MLP.evaluate(x_test, y_test, verbose=1)

# mlp_pred = MLP.predict(x_test)
# x, y = [], []

# for sample in mlp_pred:
#     x.append(sample.max() * 100)

# for confidence in x:
#     y.append(((x > confidence).sum() / len(mlp_pred)) * 100)

# plt.plot(x, y, ".", color="black", markersize=1, label="Confidence of MLP Predictions")
# plt.ylim(max(y) + 1, min(y))
# plt.xlabel("Confidence (%)")
# plt.ylabel("Higher Confidence Predictions (%)")
# plt.legend()
# plt.show()

LeNet5.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
LeNet5.fit(x=x_train, y=y_train, epochs=5)
LeNet5.evaluate(x_test, y_test, verbose=1)

# LeNet_pred = LeNet5.predict(x_test)
# x, y = [], []

# for sample in LeNet_pred:
#     x.append(sample.max() * 100)

# for confidence in x:
#     y.append(((x > confidence).sum() / len(LeNet_pred)) * 100)

# plt.plot(
#     x, y, ".", color="black", markersize=1, label="Confidence of LeNet-5 Predictions"
# )
# plt.ylim(max(y) + 1, min(y))
# plt.xlabel("Confidence (%)")
# plt.ylabel("Higher Confidence Predictions (%)")
# plt.legend()
# plt.show()

# AlexNet.compile(
#     optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
# )
# AlexNet.fit(x=x_train, y=y_train, epochs=10)
# AlexNet.evaluate(x_test, y_test)

for i in tqdm(range(len(x_test[:100]))):
	
    sample = np.array(x_test[i], ndmin=4)

    single_mlp_pred = MLP.predict(sample)

    if single_mlp_pred.max() <= 0.90:
        # print("failing to LeNet at: ", single_mlp_pred.max(), "%")
        single_LeNet_pred = LeNet5.predict(sample)

        # print("LeNet5 confidence of prediction: ", single_LeNet_pred.max(), "%")
