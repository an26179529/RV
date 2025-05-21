from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from werkzeug.utils import secure_filename
import base64
from datetime import datetime
import time
import json

# 導入.py檔案
from Detector import TFLiteDetector
from Calculate import calculate_diet_recommendations
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB最大上傳限制
app.config['MODEL_PATH'] = 'detectv2_float16.tflite'

# 如果上傳資料夾不存在，則創建
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化檢測器
detector = TFLiteDetector(
    model_path=app.config['MODEL_PATH'],
    img_size=640,
    conf_threshold=0.25,
    iou_threshold=0.45
)

# 從JSON檔案載入營養數據
with open('nutrition_data.json', 'r') as file:
    nutrition_data = json.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # 儲存上傳的檔案
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = secure_filename(f"{timestamp}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 使用OpenCV讀取圖像
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': '無法處理圖像'})
        
        # 執行檢測
        start_time = time.time()
        detections = detector.detect(img)
        inference_time = time.time() - start_time
        
        # 在圖像上繪製檢測結果
        result_img = detector.draw_detections(img, detections)
        
        # 儲存結果圖像
        result_filename = f"result_{filename}"
        result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_filepath, result_img)
        
        # 按類別計數檢測結果
        detection_counts = {}
        total_nutrition = {'calories': 0, 'protein': 0, 'carbs': 0, 'fiber': 0}
        
        for detection in detections:
            class_name = detection['class_name']
            if class_name in detection_counts:
                detection_counts[class_name] += 1
            else:
                detection_counts[class_name] = 1
            
            # 計算營養成分
            if class_name in nutrition_data:
                for nutrient, amount in nutrition_data[class_name].items():
                    total_nutrition[nutrient] += amount
        
        # 計算飲食建議
        dietary_recommendations = calculate_diet_recommendations(total_nutrition)
        
        # 準備回應資料
        result_data = {
            'detections': detection_counts,
            'result_image': f"/static/uploads/{result_filename}",
            'inference_time': f"{inference_time*1000:.2f}ms",
            'total_items': len(detections),
            'nutrition': total_nutrition,
            'recommendations': dietary_recommendations
        }
        
        return render_template('result.html', 
                              result=result_data, 
                              nutrition_data=nutrition_data,
                              timestamp=timestamp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)