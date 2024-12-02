# Классификация изображений с использованием метода K-Nearest Neighbors (KNN)
## Описание

В этом проекте была реализована модель классификации изображений с использованием метода 
**K-Nearest Neighbors (KNN)** на датасете рукописных цифр из `sklearn.datasets`.

### Этапы выполнения:

#LBL1 Импорт библиотек python

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

Вначале я импортировала все необходимые библиотеки для работы с данными, построения модели CNN и оценки ее производительности.

#LBL2 Загрузка и нормализация данных

train_data = np.load('C:/Users/Larin/Desktop/Asem/train_small.npz')
test_data = np.load('C:/Users/Larin/Desktop/Asem/test_small.npz')
X_train = train_data['data'].astype(np.float32) / 255.0
y_train = train_data['labels']
X_test = test_data['data'].astype(np.float32) / 255.0
y_test = test_data['labels']

Здесь я загрузила обучающие и тестовые данные из файлов .npz и нормализовала значения пикселей до диапазона [0, 1].

#LBL3 Аугментация данных

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)

Применяю техники аугментации данных к обучающим данным для улучшения устойчивости модели.

#LBL4 Создание модели CNN

python

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

Здесь я определила архитектуру модели CNN с использованием Keras Sequential API.

#LBL5 Компиляция модели

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

Компилирую модель с оптимизатором Adam и функцией потерь sparse categorical cross-entropy.

#LBL6 Обучение модели

history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=50, validation_data=(X_test, y_test))

Обучаю модель с использованием аугментированных данных и валидирую ее на тестовых данных.

#LBL7 Оценка точности модели

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Точность улучшенной модели CNN: {accuracy:.2f}")

Оцениваю точность модели на тестовых данных.

#LBL8 Предсказания на тестовой выборке

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

Генерирую предсказания на тестовых данных и определяю предсказанные классы.

#LBL9 Отчет классификации

print("Отчет классификации:")
print(classification_report(y_test, y_pred_classes, zero_division=0))

Печатаю отчет классификации для оценки производительности модели на каждом классе.

#LBL10 Матрица ошибок

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_classes, cmap="Blues")
plt.title("Матрица ошибок")
plt.show()

Строю матрицу ошибок для визуализации производительности модели.

## Результаты
- **Точность модели KNN** на тестовой выборке: **0.9**.

Дополнительные файлы и директории
C:/Users/Larin/Desktop/Asem/train_small.npz: Файл обучающего набора данных.
C:/Users/Larin/Desktop/Asem/test_small.npz: Файл тестового набора данных.
results.json: Файл для сохранения результатов тестирования (точность и потери).
