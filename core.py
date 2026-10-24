import os
import numpy as np
from PIL import ImageOps, Image
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam

# ГЛОБАЛЬНІ ЗМІННІ
MODEL_PATH = "model.h5"
mode = "train" #train, predict

#СТВОРЕННЯ АБО ЗАВАНТАЖЕННЯ МОДЕЛІ
if os.path.exists(MODEL_PATH):
    mode = "predict"
    print("Loading exists model...")
    model = load_model(MODEL_PATH)
else:
    mode = "train"
    print("Creating new model...")
    model = Sequential([
        Dense(128, input_shape=(784,), activation="relu"),
        Dense(256, activation="relu"),
        Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

train_images, train_labels = [], []

def train(img, number_value, train_count):
    """
    Додавання зображень в набори даних, або запуск навчання нейронної мережі.
    :param img: Одне зображення, цифри намальованої в інтерфейсі, 28x28, не інвертоване
    :param number_value: Число, яке відповідає намальованій цифрі
    :param train_count: Кількість чисел для тренування введених через інтерфейс
    :return: False - тренування не завершене, True - тренування завершене
    :rtype: bool
    :warning:
        Функцію потрібно викликати після створення моделі.
    """
    global train_images
    global train_labels
    img = ImageOps.invert(img)
    img_array = np.array(img).astype("float32")
    img_array /= 255
    img_array = img_array.reshape(784)
    train_images.append(img_array)
    train_labels.append(number_value)
    if len(train_images) >= train_count:
        print("Training")
        model.fit(np.array(train_images), np.array(train_labels) , epochs=5, verbose=0)
        model.save(MODEL_PATH)
        train_images, train_labels = [], []
        return True

def predict(img):
    """
    :param img: Одне зображення, цифри намальовані в інтерфейсі, 28x28, не інвертоване
    :return: Результат передбачення, одне із значень: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    :rtype: int
    """
    global train_images
    global train_labels
    img = ImageOps.invert(img)
    img_array = np.array(img).astype("float32")
    img_array /= 255
    img_array = img_array.reshape(1, 784)
    print("Predicting...")
    pred = model.predict(img_array)
    result = np.argmax(pred)
    return result


def _load_images_from_folder(folder):
    images = []
    labels = []
    for fname in os.listdir(folder):
        digit = int(fname.split("_", 1)[0])
        path = os.path.join(folder, fname)
        img = Image.open(path).convert("L").resize((28, 28))
        arr = np.array(img).astype("float32") / 255.0
        images.append(arr.reshape(784))
        labels.append(digit)
    return np.array(images), np.array(labels)


def train_mnist():
    global model, mode
    x, y = _load_images_from_folder("mnist/mnist/train")
    idx = np.random.permutation(len(x))
    x, y = x[idx], y[idx]
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(x, y, epochs=5, batch_size=32, validation_split=0.15, verbose=0)
    model.save(MODEL_PATH)
    mode = "predict"


def predict_mnist():
    global model
    model = load_model(MODEL_PATH)
    x, y = _load_images_from_folder("mnist/mnist/test")
    pred = model.predict(x, verbose=0)
    pred_labels = np.argmax(pred, axis=1)
    accuracy = np.mean(pred_labels == y)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    for d in range(10):
        mask = y == d
        acc_d = np.mean(pred_labels[mask] == y[mask])
        print(f"  {d}: {acc_d * 100:.1f}% ({int(mask.sum())})")
    return accuracy


def mnist():
    train_mnist()