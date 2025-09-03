import cv2
import numpy as np
from typing import List, Dict, Tuple
from ultralytics import YOLO
from parking_spaces import ParkingSpace, ParkingLot
import config

class VehicleDetector:
    def __init__(self, model_path: str = config.YOLO_MODEL):
        self.model = YOLO(model_path)
        self.confidence_threshold = config.YOLO_CONFIDENCE
        self.iou_threshold = config.YOLO_IOU_THRESHOLD
        self.vehicle_classes = self._get_vehicle_class_ids()
        
    def _get_vehicle_class_ids(self) -> List[int]:
        class_names = self.model.names
        vehicle_ids = []
        for class_id, name in class_names.items():
            if name.lower() in config.VEHICLE_CLASSES:
                vehicle_ids.append(class_id)
        return vehicle_ids
    
    def detect_vehicles(self, image: np.ndarray) -> List[Dict]:
        results = self.model(image, conf=self.confidence_threshold, iou=self.iou_threshold)
        
        vehicles = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls)
                    if class_id in self.vehicle_classes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = float(box.conf)
                        class_name = self.model.names[class_id]
                        
                        vehicles.append({
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": confidence,
                            "class": class_name
                        })
        
        return vehicles
    
    def calculate_iou(self, box1: Tuple[int, int, int, int], 
                     box2: Tuple[int, int, int, int]) -> float:
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
    
    def check_parking_occupancy(self, vehicles: List[Dict], 
                               parking_lot: ParkingLot) -> ParkingLot:
        parking_lot.reset_status()
        
        for space in parking_lot.spaces:
            space_bbox = space.get_bbox()
            max_iou = 0.0
            
            for vehicle in vehicles:
                vehicle_bbox = tuple(vehicle["bbox"])
                iou = self.calculate_iou(space_bbox, vehicle_bbox)
                max_iou = max(max_iou, iou)
                
                if iou > config.PARKING_SPACE_IOU_THRESHOLD:
                    space.status = "occupied"
                    space.confidence = vehicle["confidence"]
                    break
            
            if space.status == "empty":
                space.confidence = 1.0 - max_iou
        
        return parking_lot
    
    def draw_detections(self, image: np.ndarray, vehicles: List[Dict], 
                       parking_lot: ParkingLot) -> np.ndarray:
        result_image = image.copy()
        
        for space in parking_lot.spaces:
            color = config.OVERLAY_CONFIG["occupied_color"] if space.status == "occupied" \
                   else config.OVERLAY_CONFIG["empty_color"]
            
            x1, y1, x2, y2 = space.get_bbox()
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 
                         config.OVERLAY_CONFIG["box_thickness"])
            
            label = f"{space.id}: {space.status.upper()}"
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.putText(result_image, label, (x1, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       config.OVERLAY_CONFIG["font_scale"], 
                       color, 
                       config.OVERLAY_CONFIG["font_thickness"])
        
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle["bbox"]
            cv2.rectangle(result_image, (x1, y1), (x2, y2), 
                         (255, 0, 0), 2)
            
            label = f"{vehicle['class']}: {vehicle['confidence']:.2f}"
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (255, 0, 0), 1)
        
        summary = parking_lot.get_summary()
        status_text = f"Occupied: {summary['occupied']}/{summary['total_spaces']} | Available: {summary['available']}"
        cv2.putText(result_image, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                   (0, 0, 0), 2)
        cv2.putText(result_image, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                   (255, 255, 255), 1)
        
        return result_image