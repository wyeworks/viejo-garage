#!/usr/bin/env python3

import cv2
import json
import sys
import numpy as np
from pathlib import Path
import argparse

class ParkingSpaceCalibrator:
    def __init__(self, image_path: str):
        self.image_path = Path(image_path)
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.original_image = self.image.copy()
        self.spaces = []
        self.current_space = []
        self.drawing = False
        self.window_name = "Parking Space Calibration"
        
        # Scale image for display if too large
        self.display_scale = 1.0
        max_height = 800
        if self.image.shape[0] > max_height:
            self.display_scale = max_height / self.image.shape[0]
            new_width = int(self.image.shape[1] * self.display_scale)
            self.display_image = cv2.resize(self.image, (new_width, max_height))
        else:
            self.display_image = self.image.copy()
        
        self.original_display = self.display_image.copy()
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_space = [(x, y)]
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Update display with current rectangle
                self.display_image = self.original_display.copy()
                self.draw_existing_spaces()
                if len(self.current_space) == 1:
                    cv2.rectangle(self.display_image, self.current_space[0], 
                                (x, y), (0, 255, 255), 2)
                cv2.imshow(self.window_name, self.display_image)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if len(self.current_space) == 1:
                x1, y1 = self.current_space[0]
                x2, y2 = x, y
                
                # Ensure coordinates are in correct order
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                
                # Minimum size check
                if abs(x_max - x_min) > 20 and abs(y_max - y_min) > 20:
                    # Convert back to original image coordinates
                    orig_x = int(x_min / self.display_scale)
                    orig_y = int(y_min / self.display_scale)
                    orig_width = int((x_max - x_min) / self.display_scale)
                    orig_height = int((y_max - y_min) / self.display_scale)
                    
                    space_id = f"P{len(self.spaces) + 1}"
                    self.spaces.append({
                        "space_id": space_id,
                        "x": orig_x,
                        "y": orig_y,
                        "width": orig_width,
                        "height": orig_height
                    })
                    print(f"Added space {space_id}: x={orig_x}, y={orig_y}, "
                          f"width={orig_width}, height={orig_height}")
                    
                    self.redraw()
                    
    def draw_existing_spaces(self):
        for i, space in enumerate(self.spaces):
            # Convert to display coordinates
            x = int(space["x"] * self.display_scale)
            y = int(space["y"] * self.display_scale)
            width = int(space["width"] * self.display_scale)
            height = int(space["height"] * self.display_scale)
            
            color = (0, 255, 0)  # Green for saved spaces
            cv2.rectangle(self.display_image, (x, y), 
                        (x + width, y + height), color, 2)
            
            # Draw label
            label = space["space_id"]
            label_pos = (x + 5, y + 20)
            cv2.putText(self.display_image, label, label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def redraw(self):
        self.display_image = self.original_display.copy()
        self.draw_existing_spaces()
        cv2.imshow(self.window_name, self.display_image)
    
    def save_configuration(self):
        # Create config directory if it doesn't exist
        config_dir = Path("parking_configs")
        config_dir.mkdir(exist_ok=True)
        
        # Save configuration with image name as base
        config_name = self.image_path.stem + "_parking.json"
        config_path = config_dir / config_name
        
        config = {
            "image_path": str(self.image_path),
            "image_size": {
                "width": self.image.shape[1],
                "height": self.image.shape[0]
            },
            "spaces": self.spaces
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nConfiguration saved to: {config_path}")
        return config_path
    
    def run(self):
        print("\n" + "="*60)
        print("PARKING SPACE CALIBRATION MODE")
        print("="*60)
        print(f"Image: {self.image_path}")
        print(f"Size: {self.image.shape[1]}x{self.image.shape[0]}")
        print("\nInstructions:")
        print("- Click and drag to draw parking space rectangles")
        print("- Press 'r' to remove last space")
        print("- Press 'c' to clear all spaces")
        print("- Press 's' to save configuration")
        print("- Press 'q' or ESC to quit")
        print("="*60 + "\n")
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        cv2.imshow(self.window_name, self.display_image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # q or ESC
                print("Calibration cancelled")
                break
            elif key == ord('s'):  # Save
                if len(self.spaces) > 0:
                    config_path = self.save_configuration()
                    print(f"Saved {len(self.spaces)} parking spaces")
                    break
                else:
                    print("No spaces defined! Draw at least one space before saving.")
            elif key == ord('r'):  # Remove last
                if len(self.spaces) > 0:
                    removed = self.spaces.pop()
                    print(f"Removed space {removed['space_id']}")
                    self.redraw()
            elif key == ord('c'):  # Clear all
                if len(self.spaces) > 0:
                    self.spaces = []
                    print("Cleared all spaces")
                    self.redraw()
        
        cv2.destroyAllWindows()
        return self.spaces

def main():
    parser = argparse.ArgumentParser(description="Calibrate parking spaces for an image")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--list", action="store_true", 
                       help="List existing configurations")
    
    args = parser.parse_args()
    
    if args.list:
        config_dir = Path("parking_configs")
        if config_dir.exists():
            configs = list(config_dir.glob("*.json"))
            if configs:
                print("\nExisting configurations:")
                for config in configs:
                    with open(config, 'r') as f:
                        data = json.load(f)
                    print(f"  {config.name}: {len(data['spaces'])} spaces")
            else:
                print("\nNo configurations found")
        return
    
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    try:
        calibrator = ParkingSpaceCalibrator(args.image)
        spaces = calibrator.run()
        
        if spaces:
            print(f"\nCalibration complete! Defined {len(spaces)} parking spaces:")
            for space in spaces:
                print(f"  {space['space_id']}: x={space['x']}, y={space['y']}, "
                      f"w={space['width']}, h={space['height']}")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()