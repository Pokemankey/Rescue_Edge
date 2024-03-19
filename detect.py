
# app.py
from flask import Flask, request, jsonify
from ultralytics import YOLO
import base64
import cv2
import numpy as np
import os
from datetime import datetime
import shutil
app = Flask(__name__)

images_folder = "images"
os.makedirs(images_folder, exist_ok=True)

def ai(name):
    model = YOLO('humanFinder.pt')
    results = model.predict(f"./images/{name}", save=True, conf=0.3,classes=[0])
    for r in results:
        if len(r.boxes.xyxy) > 0:
            shutil.move(f"./runs/segment/predict/{name}", "./Predictions")

    
    shutil.rmtree("./runs")


try:
    with open('counter.txt', 'r') as file:
        x = int(file.read())
except FileNotFoundError:

    x = 1

@app.route('/upload', methods=['POST'])
def upload():
    global x  
    try:
        data = request.get_json()
        img_base64 = data['image']
        img_bytes = base64.b64decode(img_base64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Save the received image to the "images" folder with a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}.jpg"
        filepath = os.path.join(images_folder, filename)
        cv2.imwrite(filepath, img)
        ai(f"{filename}")
        x += 1
        with open('counter.txt', 'w') as file:
            file.write(str(x))

        return jsonify({'status': 'success', 'message': f'Image saved as {filename}'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    
app.run(host='0.0.0.0', port=5000)







# if _name_ == '_main_':
#     app.run(debug=True)

#     def upload():
#     try:
#         data = request.get_json()
#         img_base64 = data['image']
#         img_bytes = base64.b64decode(img_base64)
#         img_array = np.frombuffer(img_bytes, dtype=np.uint8)
#         img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

#         # Save the received image to the "images" folder with a timestamped filename
#         timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#         filename = f"{timestamp}.jpg"
#         filepath = os.path.join(images_folder, filename)
#         cv2.imwrite(filepath, img)

#         return jsonify({'status': 'success', 'message': f'Image saved as {filename}'})

#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)})
