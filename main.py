from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

# Create a folder named "images" if it doesn't exist
images_folder = "images"
os.makedirs(images_folder, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload():
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

        return jsonify({'status': 'success', 'message': f'Image saved as {filename}'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


app.run(host='0.0.0.0', port=5000)
