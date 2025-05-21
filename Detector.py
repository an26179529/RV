import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import time
from typing import List, Tuple, Dict, Any

class TFLiteDetector:
    def __init__(self, model_path: str, img_size: int = 640, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        初始化 YOLOv8 TFLite 偵測器
        
        參數:
            model_path: TFLite 模型檔案路徑
            img_size: 模型的輸入尺寸 (預設: 640，用於 YOLO)
            conf_threshold: 偵測的置信度閾值
            iou_threshold: 非最大抑制的 IoU 閾值
        """
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 載入類別名稱
        self.class_names = ['Apple', 'Banana', 'Mango', 'MiniBiscuits', 'Orange', 'Pineapple', 'Snack', 'Watermelon', 'noodles']
        self.num_classes = len(self.class_names)
        
        # 載入 TFLite 模型
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # 獲取輸入和輸出詳細資訊
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 獲取輸入形狀
        self.input_shape = self.input_details[0]['shape']
        
        # 如果與模型輸入不同，更新 img_size
        if self.input_shape[1] != self.img_size or self.input_shape[2] != self.img_size:
            self.img_size = self.input_shape[1]
    
    def preprocess_image(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int], Tuple[int, int, int, int]]:
        """
        為 YOLOv8 輸入預處理圖像
        
        參數:
            img: 原始圖像 (來自 OpenCV 的 BGR 格式)
            
        返回:
            (預處理後的圖像, 縮放因子, 原始圖像形狀, 填充資訊) 的元組
        """
        # 儲存原始圖像形狀
        original_shape = img.shape[:2]  # (高度, 寬度)
        
        # 計算縮放因子和調整大小
        h, w = original_shape
        scale = min(self.img_size / h, self.img_size / w)
        
        # 計算填充
        new_h, new_w = int(h * scale), int(w * scale)
        pad_h, pad_w = self.img_size - new_h, self.img_size - new_w
        top_pad, left_pad = pad_h // 2, pad_w // 2
        bottom_pad, right_pad = pad_h - top_pad, pad_w - left_pad
        
        # 調整圖像大小
        resized_img = cv2.resize(img, (new_w, new_h))
        
        # 創建填充後的圖像 (信箱效果)
        padded_img = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        padded_img[top_pad:top_pad+new_h, left_pad:left_pad+new_w, :] = resized_img
        
        # 轉換為 RGB 並標準化
        img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        
        # 添加批次維度
        img_input = np.expand_dims(img_norm, axis=0)
        
        return img_input, scale, original_shape, (top_pad, left_pad, bottom_pad, right_pad)
    
    def detect(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        在圖像上運行 YOLO 檢測
        
        參數:
            img: 輸入圖像 (來自 OpenCV 的 BGR 格式)
            
        返回:
            偵測結果列表
        """
        # 預處理圖像
        img_input, scale, original_shape, padding = self.preprocess_image(img)
        top_pad, left_pad, _, _ = padding
        
        # 設置輸入張量
        self.interpreter.set_tensor(self.input_details[0]['index'], img_input)
        
        # 運行推論
        start_time = time.time()
        self.interpreter.invoke()
        inference_time = time.time() - start_time
        
        # 獲取輸出
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # 處理 YOLOv8 輸出，形狀為 (1, 13, 8400)
        # 對於這種特定形狀:
        # - 前 4 行 (0-3) 是邊界框座標 (x, y, w, h)
        # - 後 9 行 (4-12) 是 9 個類別的概率
        
        if output.shape[1] == 13 and output.shape[2] == 8400:
            # 轉置以獲得形狀 (1, 8400, 13)
            output = np.transpose(output, (0, 2, 1))
            
            # 提取邊界框和分數
            boxes = output[0, :, 0:4]  # [x_中心, y_中心, 寬度, 高度]
            scores = output[0, :, 4:]  # 類別分數
            
            # 獲取每個框的最佳類別和分數
            class_scores = np.max(scores, axis=1)
            class_ids = np.argmax(scores, axis=1)
            
            # 根據置信度閾值過濾
            keep_indices = class_scores > self.conf_threshold
            
            if np.any(keep_indices):
                boxes = boxes[keep_indices]
                class_scores = class_scores[keep_indices]
                class_ids = class_ids[keep_indices]
            else:
                return []
            
            # 將標準化的邊界框座標轉換為像素座標
            detections = []
            h, w = original_shape
            scale_factor = min(self.img_size / h, self.img_size / w)
            
            for i in range(len(boxes)):
                # 獲取邊界框座標 (標準化為輸入大小)
                x_center, y_center, width, height = boxes[i]
                
                # 轉換為填充後圖像上的像素
                x_min = int((x_center - width / 2) * self.img_size)
                y_min = int((y_center - height / 2) * self.img_size)
                x_max = int((x_center + width / 2) * self.img_size)
                y_max = int((y_center + height / 2) * self.img_size)
                
                # 移除填充
                x_min = x_min - left_pad
                y_min = y_min - top_pad
                x_max = x_max - left_pad
                y_max = y_max - top_pad
                
                # 限制在調整大小後的圖像範圍內
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                x_min = max(0, min(x_min, new_w - 1))
                y_min = max(0, min(y_min, new_h - 1))
                x_max = max(0, min(x_max, new_w - 1))
                y_max = max(0, min(y_max, new_h - 1))
                
                # 轉換回原始圖像尺寸
                x_min = int(x_min / scale_factor)
                y_min = int(y_min / scale_factor)
                x_max = int(x_max / scale_factor)
                y_max = int(y_max / scale_factor)
                
                # 將檢測添加到列表
                detections.append({
                    'class_id': int(class_ids[i]),
                    'class_name': self.class_names[int(class_ids[i])],
                    'confidence': float(class_scores[i]),
                    'bbox': [x_min, y_min, x_max, y_max]
                })
            
            # 應用非最大抑制
            return self._non_max_suppression(detections)
        else:
            return []
    
    def _non_max_suppression(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        應用非最大抑制來過濾重複的檢測結果
        
        參數:
            detections: 檢測結果字典列表
            
        返回:
            過濾後的檢測結果列表
        """
        if len(detections) == 0:
            return []
        
        # 按類別分組檢測結果
        class_groups = {}
        for detection in detections:
            class_id = detection['class_id']
            if class_id not in class_groups:
                class_groups[class_id] = []
            class_groups[class_id].append(detection)
        
        # 對每個類別應用 NMS
        nms_results = []
        for class_id, group in class_groups.items():
            # 按置信度排序
            group = sorted(group, key=lambda x: x['confidence'], reverse=True)
            
            while len(group) > 0:
                # 取置信度最高的檢測結果
                best_detection = group.pop(0)
                nms_results.append(best_detection)
                
                # 過濾掉高 IoU 的檢測結果
                group = [d for d in group if self._calculate_iou(best_detection['bbox'], d['bbox']) < self.iou_threshold]
        
        return nms_results
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        計算兩個框之間的 IoU
        
        參數:
            box1: 第一個框 [x1, y1, x2, y2]
            box2: 第二個框 [x1, y1, x2, y2]
            
        返回:
            IoU 值
        """
        # 計算交集面積
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        # 計算聯集面積
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        # 計算 IoU
        if union_area == 0:
            return 0
        return intersection_area / union_area
    
    def draw_detections(self, img: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        在圖像上繪製檢測結果
        
        參數:
            img: 輸入圖像
            detections: 檢測結果字典列表
            
        返回:
            繪製了檢測結果的圖像
        """
        img_copy = img.copy()
        
        # 為每個類別定義顏色映射，以確保顏色一致性
        color_map = {}
        for i, name in enumerate(self.class_names):
            color_map[i] = (
                int(120 + (i * 30) % 135),  # B
                int(50 + (i * 40) % 205),   # G 
                int(50 + (i * 60) % 205)    # R
            )
        
        for detection in detections:
            # 提取資訊
            class_id = detection['class_id']
            class_name = detection['class_name']
            confidence = detection['confidence']
            x1, y1, x2, y2 = detection['bbox']
            
            # 獲取此類別的顏色
            color = color_map.get(class_id, (0, 255, 0))
            
            # 繪製邊界框
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            # 準備標籤文字
            label = f"{class_name} {confidence:.2f}"
            
            # 獲取文字尺寸
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # 繪製標籤背景
            cv2.rectangle(
                img_copy,
                (x1, y1 - label_height - baseline - 5),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            # 繪製標籤文字
            cv2.putText(
                img_copy,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        return img_copy