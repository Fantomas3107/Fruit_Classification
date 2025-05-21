import os
import argparse

# Импорт модулей проекта
from config.settings import (
    IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    DATA_DIR, MODEL_SAVE_PATH, BEST_MODEL_PATH, AUGMENTATION_CONFIG
)
from data.data_loader import download_dataset, create_data_generators
from models.cnn_model import create_custom_cnn, compile_model
from training.trainer import train_model, load_best_model
from utils.visualization import (
    plot_training_history, plot_model_architecture,
    evaluate_model, show_prediction_examples
)
from prediction.predictor import predict_image


def parse_args():
    parser = argparse.ArgumentParser(description='Система определения свежести фруктов и овощей')
    
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'predict'],
                        help='Режим работы: обучение, оценка или предсказание')
    
    parser.add_argument('--image_path', type=str, default=None,
                        help='Путь к изображению для предсказания')
    
    parser.add_argument('--download', action='store_true',
                        help='Загрузить датасет перед запуском')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.download:
        data_path = download_dataset()
        print(f"Датасет загружен: {data_path}")
    
    train_generator, validation_generator, class_names = create_data_generators(
        DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, AUGMENTATION_CONFIG
    )
    
    if args.mode == 'train':
        print("Запуск режима обучения...")
        
        model = create_custom_cnn(IMG_HEIGHT, IMG_WIDTH, len(class_names))
        model = compile_model(model, LEARNING_RATE)

        plot_model_architecture(model)

        model, history = train_model(
            train_generator, 
            validation_generator, 
            IMG_HEIGHT, 
            IMG_WIDTH,
            EPOCHS, 
            LEARNING_RATE, 
            BEST_MODEL_PATH, 
            class_names
        )

        plot_training_history(history)

        model = load_best_model(BEST_MODEL_PATH)

        model.save(MODEL_SAVE_PATH)
        print(f"Модель сохранена как '{MODEL_SAVE_PATH}'")

        evaluate_model(model, validation_generator, class_names)
 
        show_prediction_examples(validation_generator, model, class_names)
        
    elif args.mode == 'eval':
        print("Запуск режима оценки...")    
        model = load_best_model(MODEL_SAVE_PATH)
        
        val_loss, val_acc = model.evaluate(validation_generator)
        print(f"Точность на валидационных данных: {val_acc*100:.2f}%")
        
        evaluate_model(model, validation_generator, class_names)
        
        show_prediction_examples(validation_generator, model, class_names)
        
    elif args.mode == 'predict':
        if args.image_path is None:
            print("Ошибка: не указан путь к изображению для предсказания")
            return
        
        print(f"Запуск режима предсказания для изображения: {args.image_path}")
        
        model = load_best_model(MODEL_SAVE_PATH)
        
        result = predict_image(args.image_path, model, class_names, IMG_HEIGHT, IMG_WIDTH)
        
        print(f"Предсказанный класс: {result['class']}")
        print(f"Уверенность: {result['confidence']:.2f}")
        print("Все предсказания:")
        for class_name, confidence in result['predictions'].items():
            print(f"  {class_name}: {confidence:.4f}")


if __name__ == "__main__":
    main()