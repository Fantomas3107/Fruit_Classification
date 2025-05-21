import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf


def plot_training_history(history):
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


def plot_model_architecture(model):
    tf.keras.utils.plot_model(
        model, 
        to_file='model_architecture.png', 
        show_shapes=True, 
        show_dtype=True, 
        show_layer_names=True
    )


def evaluate_model(model, validation_generator, class_names):
    validation_generator.reset()
    validation_steps = validation_generator.samples // validation_generator.batch_size
    
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