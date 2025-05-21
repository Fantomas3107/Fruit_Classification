import numpy as np
import tensorflow as tf


def predict_image(image_path, model, class_names, img_height, img_width):
    img = tf.keras.preprocessing.image.load_img(
        image_path, 
        target_size=(img_height, img_width)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)

    prediction = model.predict(img_array)[0]
    predicted_class = np.argmax(prediction)

    result = {
        'class': class_names[predicted_class],
        'confidence': float(prediction[predicted_class]),
        'predictions': {class_names[i]: float(prediction[i]) for i in range(len(class_names))}
    }
    
    return result


def predict_batch(image_paths, model, class_names, img_height, img_width):
    results = []
    
    for image_path in image_paths:
        result = predict_image(image_path, model, class_names, img_height, img_width)
        results.append(result)
    
    return results