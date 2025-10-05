import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

def process_image(image_np, target_size=(224, 224)):
    """يعالج صورة NumPy لتكون جاهزة للنموذج."""
    image = tf.convert_to_tensor(image_np, dtype=tf.float32)
    image = tf.image.resize(image, target_size)
    image /= 255.0 
    return image.numpy()

def predict(image_path, model, top_k):
    """يتنبأ بتصنيف الصورة ويعيد أعلى K احتمالات وفهارس."""
    try:
        im = Image.open(image_path)
        image_np = np.asarray(im)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None, None

    processed_image = process_image(image_np)
    expanded_image = np.expand_dims(processed_image, axis=0)
    predictions = model.predict(expanded_image)
    probs, indices = tf.math.top_k(predictions, k=top_k)
    return probs.numpy()[0], indices.numpy()[0]

def main():
    parser = argparse.ArgumentParser(description="Predict flower name from an image.")
    parser.add_argument('image_path', help="Path to the input image.")
    parser.add_argument('model_path', help="Path to the trained Keras model (.h5 file).")
    parser.add_argument('--top_k', type=int, default=1, help="Return top K most likely classes.")
    parser.add_argument('--category_names', type=str, help="Path to a JSON file mapping categories to real names.")
    parser.add_argument('--gpu', action='store_true', help="Use GPU for inference if available.")
    
    args = parser.parse_args()

    if not args.gpu:
        tf.config.set_visible_devices([], 'GPU')
    else:
        if not tf.config.list_physical_devices('GPU'):
            print("GPU requested but not available. Running on CPU.")

    try:
        model = tf.keras.models.load_model(
            args.model_path,
            custom_objects={'KerasLayer': hub.KerasLayer}
        )
    except (FileNotFoundError, IOError):
        print(f"Error: Model file not found at {args.model_path}")
        return
        
    probs, indices = predict(args.image_path, model, args.top_k)
    
    if probs is None:
        return

    class_names = None
    if args.category_names:
        try:
            with open(args.category_names, 'r') as f:
                class_names = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Category names file not found at {args.category_names}. Displaying indices instead.")
            return

    print(f"\nTop {args.top_k} Predictions:")
    print("-" * 30)

    for i in range(args.top_k):
        prob = probs[i]
        class_index = indices[i] 
        
        if class_names:
            
            lookup_key = str(class_index + 1)
            name = class_names.get(lookup_key, f"Unknown Index: {lookup_key}")
        else:
            name = f"Index: {class_index}"
            
        print(f"{i+1}: {name:<25} | Probability: {prob:.4f}")

if __name__ == '__main__':
    main()