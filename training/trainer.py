import tensorflow as tf
from models.cnn_model import create_custom_cnn, compile_model, create_callbacks


def train_model(train_generator, validation_generator, img_height, img_width, 
                 epochs, learning_rate, best_model_path, class_names=None):
    
    num_classes = len(train_generator.class_indices)

    model = create_custom_cnn(img_height, img_width, num_classes)
    model = compile_model(model, learning_rate)

    model.summary()

    callbacks = create_callbacks(best_model_path)

    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = validation_generator.samples // validation_generator.batch_size
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history


def load_best_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model