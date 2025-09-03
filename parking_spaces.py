from typing import List, Dict, Tuple
import json
from pathlib import Path

class ParkingSpace:
    def __init__(self, space_id: str, x: int, y: int, width: int, height: int):
        self.id = space_id
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.status = "empty"
        self.confidence = 0.0
        
    def get_bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def get_center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "status": self.status,
            "confidence": self.confidence
        }

class ParkingLot:
    def __init__(self, image_path: str = None):
        self.spaces: List[ParkingSpace] = []
        
        # Try to load configuration for specific image
        if image_path:
            config_loaded = self.load_image_config(image_path)
            if not config_loaded:
                self.load_default_spaces()
        else:
            self.load_default_spaces()
    
    def load_image_config(self, image_path: str) -> bool:
        """Load parking configuration specific to an image"""
        from pathlib import Path
        
        image_path = Path(image_path)
        config_dir = Path("parking_configs")
        config_file = config_dir / f"{image_path.stem}_parking.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                self.spaces.clear()
                for space_config in config["spaces"]:
                    space = ParkingSpace(**space_config)
                    self.spaces.append(space)
                
                print(f"Loaded {len(self.spaces)} parking spaces from {config_file.name}")
                return True
            except Exception as e:
                print(f"Error loading config {config_file}: {e}")
                return False
        return False
    
    def load_default_spaces(self):
        # Two parking spaces inside the garage (visible through the gates)
        # Based on detected car at [543, 1328, 1939, 2679] and [1701, 648, 3022, 1939]
        # The garage spaces are in the lower portion of the image
        default_spaces = [
            {"space_id": "G1", "x": 500, "y": 1300, "width": 1000, "height": 1400},  # Left garage space
            {"space_id": "G2", "x": 1600, "y": 1300, "width": 1000, "height": 1400}, # Right garage space
        ]
        
        for space_config in default_spaces:
            space = ParkingSpace(**space_config)
            self.spaces.append(space)
    
    def load_from_file(self, filepath: Path):
        with open(filepath, 'r') as f:
            spaces_data = json.load(f)
        
        self.spaces.clear()
        for space_config in spaces_data:
            space = ParkingSpace(**space_config)
            self.spaces.append(space)
    
    def save_to_file(self, filepath: Path):
        spaces_data = [space.to_dict() for space in self.spaces]
        with open(filepath, 'w') as f:
            json.dump(spaces_data, f, indent=2)
    
    def reset_status(self):
        for space in self.spaces:
            space.status = "empty"
            space.confidence = 0.0
    
    def get_space_by_id(self, space_id: str) -> ParkingSpace:
        for space in self.spaces:
            if space.id == space_id:
                return space
        return None
    
    def get_summary(self) -> Dict:
        total = len(self.spaces)
        occupied = sum(1 for s in self.spaces if s.status == "occupied")
        available = total - occupied
        
        return {
            "total_spaces": total,
            "occupied": occupied,
            "available": available,
            "occupancy_rate": occupied / total if total > 0 else 0,
            "available_spaces": [s.id for s in self.spaces if s.status == "empty"],
            "occupied_spaces": [s.id for s in self.spaces if s.status == "occupied"]
        }