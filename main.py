# Импортируем необходимые библиотеки
import os
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import kagglehub

path = kagglehub.dataset_download("raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables")
print("Путь к файлам датасета:", path)

data_dir = "/kaggle/input/fresh-and-stale-images-of-fruits-and-vegetables"

img_height = 224
img_width = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)
print(f"Количество классов: {num_classes}")
print(f"Классы: {class_names}")

def create_custom_cnn():
    model = models.Sequential()

    model.add(layers.Input(shape=(img_height, img_width, 3)))

    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1'))
    model.add(layers.BatchNormalization(name='bn1'))
    model.add(layers.MaxPooling2D((2, 2), name='pool1'))

    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2'))
    model.add(layers.BatchNormalization(name='bn2'))
    model.add(layers.MaxPooling2D((2, 2), name='pool2'))

    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3'))
    model.add(layers.BatchNormalization(name='bn3'))
    model.add(layers.MaxPooling2D((2, 2), name='pool3'))

    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4'))
    model.add(layers.BatchNormalization(name='bn4'))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv5'))
    model.add(layers.BatchNormalization(name='bn5'))
    model.add(layers.MaxPooling2D((2, 2), name='pool4'))

    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv6'))
    model.add(layers.BatchNormalization(name='bn6'))
    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv7'))
    model.add(layers.BatchNormalization(name='bn7'))
    model.add(layers.MaxPooling2D((2, 2), name='pool5'))

    model.add(layers.GlobalAveragePooling2D(name='gap'))
    model.add(layers.Dense(512, activation='relu', name='fc1'))
    model.add(layers.Dropout(0.5, name='dropout1'))
    model.add(layers.Dense(256, activation='relu', name='fc2'))
    model.add(layers.Dropout(0.3, name='dropout2'))
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))

    return model

model = create_custom_cnn()

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

tf.keras.utils.plot_model(model, to_file='model_architecture.png', show_shapes=True, show_dtype=True, show_layer_names=True)

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

epochs = 30
steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

model = tf.keras.models.load_model('best_model.h5')

val_loss, val_acc = model.evaluate(validation_generator)
print(f"Точность на валидационных данных: {val_acc*100:.2f}%")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Точность на обучении')
plt.plot(history.history['val_accuracy'], label='Точность на валидации')
plt.title('Динамика точности модели')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Потери на обучении')
plt.plot(history.history['val_loss'], label='Потери на валидации')
plt.title('Динамика функции потерь')
plt.xlabel('Эпоха')
plt.ylabel('Значение функции потерь')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

validation_generator.reset()
y_pred = []
y_true = []

for i in range(validation_steps):
    x_batch, y_batch = next(validation_generator)
    y_batch_pred = model.predict(x_batch)
    y_pred.extend(np.argmax(y_batch_pred, axis=1))
    y_true.extend(np.argmax(y_batch, axis=1))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Матрица ошибок')
plt.ylabel('Истинный класс')
plt.xlabel('Предсказанный класс')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

print("Отчет о классификации:")
print(classification_report(y_true, y_pred, target_names=class_names))

def show_prediction_examples(generator, model, class_names, num_examples=5):
    generator.reset()
    plt.figure(figsize=(15, num_examples * 3))

    for i in range(num_examples):
        x_batch, y_batch = next(generator)
        img = x_batch[0]
        true_label = np.argmax(y_batch[0])

        prediction = model.predict(np.expand_dims(img, axis=0))[0]
        pred_label = np.argmax(prediction)

        plt.subplot(num_examples, 1, i+1)
        plt.imshow(img)

        title_color = 'green' if pred_label == true_label else 'red'
        title = f"Истинный: {class_names[true_label]}\nПредсказанный: {class_names[pred_label]} ({prediction[pred_label]:.2f})"
        plt.title(title, color=title_color)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('prediction_examples.png')
    plt.show()

show_prediction_examples(validation_generator, model, class_names)

model.save('fruit_freshness_model.h5')
print("Модель сохранена как 'fruit_freshness_model.h5'")

def predict_image(image_path, model, class_names):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)

    prediction = model.predict(img_array)[0]
    predicted_class = np.argmax(prediction)

    return {
        'class': class_names[predicted_class],
        'confidence': float(prediction[predicted_class]),
        'predictions': {class_names[i]: float(prediction[i]) for i in range(len(class_names))}
    }

# Пример использования
# result = predict_image('path_to_your_image.jpg', model, class_names)
# print(result)