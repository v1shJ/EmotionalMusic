import cv2
import numpy as np

def predict_emotion_from_image(image_path, model):
    EMOTION_LABELS = {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'sadness', 6: 'surprise'}
    TARGET_SIZE = (48, 48)
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return "Error: Could not load image."
    
    resized_img = cv2.resize(img, TARGET_SIZE)
    processed_img = np.expand_dims(resized_img, axis=-1)
    processed_img = processed_img / 255.0
    input_batch = np.expand_dims(processed_img, axis=0)

    # Assuming 'model' is loaded globally (e.g., loaded_model = load_model('ck_fer_model.keras'))
    predictions = model.predict(input_batch, verbose=0)
    
    predicted_class_index = np.argmax(predictions)
    predicted_emotion = EMOTION_LABELS[predicted_class_index]
    
    return predicted_emotion


if __name__ == "__main__":
    test_image_path = '../TestingImages/neutral.jpeg'  
    result = predict_emotion_from_image(test_image_path)
    print(result)