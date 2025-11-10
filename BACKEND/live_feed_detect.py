import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

model = load_model('ck_fer_model.keras')

TARGET_SIZE = (48, 48)
EMOTION_LABELS = {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'sadness', 6: 'surprise'}
INFERENCE_INTERVAL = 5

def live_emotion_inference():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    last_inference_time = time.time()
    current_emotion = "---"
    current_confidence = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        if current_time - last_inference_time >= INFERENCE_INTERVAL:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(gray_frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            processed_img = np.expand_dims(resized_img, axis=-1)
            processed_img = processed_img / 255.0
            input_batch = np.expand_dims(processed_img, axis=0)
            predictions = model.predict(input_batch, verbose=0)
            predicted_class_index = np.argmax(predictions)
            current_emotion = EMOTION_LABELS[predicted_class_index]
            current_confidence = np.max(predictions)
            last_inference_time = current_time

        cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, 70), (0, 0, 0), -1)
        display_text = f"Emotion: {current_emotion} | Conf: {current_confidence:.2f}"
        cv2.putText(frame, display_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        time_to_next = INFERENCE_INTERVAL - (current_time - last_inference_time)
        cv2.putText(frame, f"Next in: {time_to_next:.1f}s", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Live Facial Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    live_emotion_inference()