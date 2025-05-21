import os
import kagglehub
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def download_dataset():

    path = kagglehub.dataset_download("raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables")
    print("Путь к файлам датасета:", path)
    return path


def create_data_generators(data_dir, img_height, img_width, batch_size, augmentation_config):

    train_datagen = ImageDataGenerator(**augmentation_config)
    
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
    
    return train_generator, validation_generator, class_names