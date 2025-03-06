from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import joblib  # or tensorflow.keras.models.load_model if using deep learning

app = Flask(__name__)

# Load the trained model (update this based on your model type)
model = joblib.load("model.pkl")  # Change this if using TensorFlow/Keras

# Start the webcam
camera = cv2.VideoCapture(0)  # 0 for default webcam, change if using an external camera

def generate_frames():
    while True:
        success, frame = camera.read()  # Capture frame from the webcam
        if not success:
            break
        else:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Yield frame as a response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webcam')
def webcam():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Capture frame from webcam
        success, frame = camera.read()
        if not success:
            return jsonify({'error': 'Failed to capture image'})

        # Preprocess frame (resize, grayscale, normalization, etc.)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale (if needed)
        frame = cv2.resize(frame, (64, 64))  # Resize as per model input shape
        frame = frame / 255.0  # Normalize if required
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(frame)

        # Convert prediction to text label
        predicted_class = np.argmax(prediction)  # Assuming classification model
        labels = ["Hello", "Thank You", "Yes", "No", "I Love You"]  # Example labels
        predicted_text = labels[predicted_class]

        return jsonify({'prediction': predicted_text})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
