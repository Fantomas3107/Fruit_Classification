import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def create_custom_cnn(img_height, img_width, num_classes):

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


def compile_model(model, learning_rate=0.001):

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_callbacks(best_model_path):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        best_model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    return [checkpoint, early_stopping, reduce_lr]