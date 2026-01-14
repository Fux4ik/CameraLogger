import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import time
from flask import Flask, Response
import RPi.GPIO as GPIO

LED_PINS = [17, 27, 22]

GPIO.setmode(GPIO.BCM)
for pin in LED_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

print("Loading model...")
interpreter = tf.lite.Interpreter(model_path="person_detector.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape: {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")

camera = cv2.VideoCapture(0)

print("\nStarting detection...")
print("Open http://172.20.10.2:5000 in browser to view stream")

last_detection = None

app = Flask(__name__)

def generate_frames():
    global last_detection
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (250, 200))
        input_data = resized.reshape(1, 200, 250, 1).astype(np.float32) / 255.0
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        prediction = output[0][0]
        is_person = prediction > 0.5

        # LED control based on confidence
        if is_person:  
            if prediction < 0.66:
                num_leds = 0
            elif prediction < 0.83:
                num_leds = 1
            else:
                num_leds = 2
        
            for i in range(3):
                if i <= num_leds:
                    GPIO.output(LED_PINS[i], GPIO.LOW)
                else:
                    GPIO.output(LED_PINS[i], GPIO.HIGH)
        else:
            for pin in LED_PINS:
                GPIO.output(pin, GPIO.HIGH)
        
        label = f"Person: {prediction:.2f}" if is_person else f"No person: {prediction:.2f}"
        color = (0, 255, 0) if is_person else (0, 0, 255)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        if is_person and last_detection != "person":
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Person detected! (confidence: {prediction:.2f})")
            with open('detections.log', 'a') as f:
                f.write(f"{timestamp} - Person entered (conf: {prediction:.2f})\n")
            last_detection = "person"
        elif not is_person and last_detection == "person":
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Person left (confidence: {prediction:.2f})")
            with open('detections.log', 'a') as f:
                f.write(f"{timestamp} - Person left (conf: {prediction:.2f})\n")
            last_detection = "no_person"
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return '<html><body><h1>Person Detection</h1><img src="/video" width="640"></body></html>'

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        GPIO.cleanup()
        camera.release()
