#!/usr/bin/env python3

import cv2
import numpy as np
from typing import List, Dict, Tuple
from ultralytics import YOLO
from parking_spaces import ParkingSpace, ParkingLot
import config

class EnhancedVehicleDetector:
    """Enhanced detection with image pre-processing for occluded vehicles"""
    
    def __init__(self, model_path: str = config.YOLO_MODEL):
        self.model = YOLO(model_path)
        self.confidence_threshold = 0.1  # Even lower threshold
        self.iou_threshold = config.YOLO_IOU_THRESHOLD
        self.vehicle_classes = self._get_vehicle_class_ids()
        
    def _get_vehicle_class_ids(self) -> List[int]:
        class_names = self.model.names
        vehicle_ids = []
        for class_id, name in class_names.items():
            if name.lower() in config.VEHICLE_CLASSES:
                vehicle_ids.append(class_id)
        return vehicle_ids
    
    def preprocess_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply multiple pre-processing techniques"""
        processed_images = {
            'original': image,
            'enhanced': self.enhance_contrast_clahe(image),
            'bright': self.adjust_brightness_contrast(image, 1.3, 20),
            'edges': self.enhance_edges(image),
            'denoised': self.denoise_image(image),
            'gate_removed': self.remove_horizontal_lines(image)
        }
        return processed_images
    
    def enhance_contrast_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge and convert back to BGR
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced
    
    def adjust_brightness_contrast(self, image: np.ndarray, 
                                  alpha: float = 1.2, beta: int = 10) -> np.ndarray:
        """Adjust brightness and contrast"""
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted
    
    def enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """Enhance edges to make vehicles more visible"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to make them thicker
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Convert back to 3-channel
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Combine with original
        result = cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)
        return result
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Remove noise while preserving edges"""
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        return denoised
    
    def remove_horizontal_lines(self, image: np.ndarray) -> np.ndarray:
        """Try to reduce the impact of horizontal gate bars"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detected_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Create mask of lines
        _, line_mask = cv2.threshold(detected_lines, 50, 255, cv2.THRESH_BINARY)
        line_mask = cv2.dilate(line_mask, np.ones((3,1), np.uint8), iterations=1)
        
        # Inpaint the lines
        result = cv2.inpaint(image, line_mask, 3, cv2.INPAINT_TELEA)
        return result
    
    def detect_vehicles_multi(self, image: np.ndarray) -> List[Dict]:
        """Run detection on multiple processed versions and combine results"""
        
        # Preprocess image with multiple techniques
        processed_images = self.preprocess_image(image)
        
        all_detections = []
        detection_counts = {}
        
        # Run detection on each processed version
        for proc_name, proc_image in processed_images.items():
            print(f"  Detecting on {proc_name} version...")
            results = self.model(proc_image, conf=self.confidence_threshold, iou=self.iou_threshold)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        if class_id in self.vehicle_classes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            confidence = float(box.conf)
                            class_name = self.model.names[class_id]
                            
                            detection = {
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "confidence": confidence,
                                "class": class_name,
                                "source": proc_name
                            }
                            all_detections.append(detection)
                            
                            # Track which processing methods found vehicles
                            bbox_key = f"{int(x1/50)},{int(y1/50)}"  # Grid-based clustering
                            if bbox_key not in detection_counts:
                                detection_counts[bbox_key] = []
                            detection_counts[bbox_key].append(detection)
        
        # Combine detections using NMS and voting
        combined_detections = self.combine_detections(all_detections, detection_counts)
        
        print(f"  Total detections across all versions: {len(all_detections)}")
        print(f"  Combined detections after voting: {len(combined_detections)}")
        
        return combined_detections
    
    def combine_detections(self, all_detections: List[Dict], 
                          detection_counts: Dict) -> List[Dict]:
        """Combine detections from multiple sources using voting"""
        
        final_detections = []
        
        # For each cluster of detections
        for bbox_key, detections in detection_counts.items():
            if len(detections) >= 2:  # At least 2 methods agree
                # Use the detection with highest confidence
                best = max(detections, key=lambda x: x['confidence'])
                
                # Boost confidence based on agreement
                best['confidence'] = min(0.9, best['confidence'] * (1 + 0.1 * len(detections)))
                best['votes'] = len(detections)
                final_detections.append(best)
            elif len(detections) == 1 and detections[0]['confidence'] > 0.3:
                # Single detection with reasonable confidence
                final_detections.append(detections[0])
        
        return final_detections
    
    def detect_by_color(self, image: np.ndarray, parking_lot: ParkingLot) -> List[Dict]:
        """Detect vehicles by color in parking spaces"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        color_detections = []
        
        for space in parking_lot.spaces:
            x1, y1, x2, y2 = space.get_bbox()
            roi = hsv[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue
            
            # Check for car-like colors (excluding road/concrete colors)
            # Blue car range
            blue_lower = np.array([100, 50, 50])
            blue_upper = np.array([130, 255, 255])
            blue_mask = cv2.inRange(roi, blue_lower, blue_upper)
            
            # Silver/Gray car range
            gray_lower = np.array([0, 0, 100])
            gray_upper = np.array([180, 30, 200])
            gray_mask = cv2.inRange(roi, gray_lower, gray_upper)
            
            # White car range
            white_lower = np.array([0, 0, 200])
            white_upper = np.array([180, 30, 255])
            white_mask = cv2.inRange(roi, white_lower, white_upper)
            
            # Black car range
            black_lower = np.array([0, 0, 0])
            black_upper = np.array([180, 255, 50])
            black_mask = cv2.inRange(roi, black_lower, black_upper)
            
            # Combine masks
            car_mask = blue_mask | gray_mask | white_mask | black_mask
            
            # Calculate percentage of car-like colors
            car_pixels = cv2.countNonZero(car_mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            car_ratio = car_pixels / total_pixels if total_pixels > 0 else 0
            
            # If significant car-like colors detected
            if car_ratio > 0.15:  # At least 15% of space has car colors
                color_detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": min(0.7, car_ratio),
                    "class": "car",
                    "source": "color_detection",
                    "space_id": space.id
                })
                print(f"  Color detection in {space.id}: {car_ratio:.2%} car-like colors")
        
        return color_detections
    
    def check_parking_occupancy_enhanced(self, image: np.ndarray,
                                        parking_lot: ParkingLot) -> ParkingLot:
        """Enhanced parking occupancy check with multiple detection methods"""
        
        parking_lot.reset_status()
        
        # Method 1: Multi-processed YOLO detection
        vehicles = self.detect_vehicles_multi(image)
        
        # Method 2: Color-based detection as backup
        color_vehicles = self.detect_by_color(image, parking_lot)
        
        # Combine all detections
        all_vehicles = vehicles + color_vehicles
        
        print(f"\n  Combined detections: {len(vehicles)} from YOLO, {len(color_vehicles)} from color")
        
        # Check occupancy for each space
        for space in parking_lot.spaces:
            space_bbox = space.get_bbox()
            max_iou = 0.0
            best_confidence = 0.0
            
            for vehicle in all_vehicles:
                vehicle_bbox = tuple(vehicle["bbox"])
                iou = self.calculate_iou(space_bbox, vehicle_bbox)
                
                if iou > config.PARKING_SPACE_IOU_THRESHOLD:
                    space.status = "occupied"
                    space.confidence = max(best_confidence, vehicle["confidence"])
                    best_confidence = space.confidence
                    print(f"    {space.id} occupied by {vehicle['source']} detection")
                    break
                
                max_iou = max(max_iou, iou)
            
            if space.status == "empty":
                space.confidence = 1.0 - max_iou
        
        return parking_lot
    
    def calculate_iou(self, box1: Tuple[int, int, int, int], 
                     box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        intersect_xmin = max(x1_min, x2_min)
        intersect_ymin = max(y1_min, y2_min)
        intersect_xmax = min(x1_max, x2_max)
        intersect_ymax = min(y1_max, y2_max)
        
        if intersect_xmax < intersect_xmin or intersect_ymax < intersect_ymin:
            return 0.0
        
        intersect_area = (intersect_xmax - intersect_xmin) * (intersect_ymax - intersect_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - intersect_area
        
        if union_area == 0:
            return 0.0
        
        return intersect_area / union_area