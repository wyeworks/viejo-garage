#!/usr/bin/env python3

import sys
import json
from pathlib import Path
import cv2
import argparse
from datetime import datetime
import config
from enhanced_detector import EnhancedVehicleDetector
from parking_spaces import ParkingLot
from detector import VehicleDetector

def process_image(image_path: Path, save_output: bool = True) -> dict:
    print(f"\n{'='*60}")
    print(f"Processing: {image_path.name}")
    print(f"{'='*60}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    print(f"Image dimensions: {image.shape[1]}x{image.shape[0]}")
    
    # Initialize parking lot with image-specific configuration
    parking_lot = ParkingLot(str(image_path))
    
    print(f"\nUsing Enhanced Detection with pre-processing...")
    print(f"Processing {len(parking_lot.spaces)} parking spaces...")
    
    # Use enhanced detector
    detector = EnhancedVehicleDetector()
    parking_lot = detector.check_parking_occupancy_enhanced(image, parking_lot)
    
    # Get summary
    summary = parking_lot.get_summary()
    print(f"\nParking Status:")
    print(f"  Total spaces: {summary['total_spaces']}")
    print(f"  Occupied: {summary['occupied']}")
    print(f"  Available: {summary['available']}")
    print(f"  Occupancy rate: {summary['occupancy_rate']:.1%}")
    
    if summary['available_spaces']:
        print(f"  Available spaces: {', '.join(summary['available_spaces'])}")
    if summary['occupied_spaces']:
        print(f"  Occupied spaces: {', '.join(summary['occupied_spaces'])}")
    
    # Draw detections on image using regular detector for visualization
    regular_detector = VehicleDetector()
    result_image = regular_detector.draw_detections(image, [], parking_lot)
    
    # Save output if requested
    if save_output:
        output_path = config.OUTPUT_DIR / f"{image_path.stem}_enhanced_result.jpg"
        cv2.imwrite(str(output_path), result_image)
        print(f"\nOutput saved to: {output_path}")
        
        # Save processed versions for debugging
        processed = detector.preprocess_image(image)
        for name, proc_img in processed.items():
            if name != 'original':
                debug_path = config.OUTPUT_DIR / f"{image_path.stem}_{name}.jpg"
                cv2.imwrite(str(debug_path), proc_img)
                print(f"Debug image saved: {debug_path.name}")
        
        # Save JSON output
        json_output = {
            "image": str(image_path),
            "timestamp": datetime.now().isoformat(),
            "detection_mode": "enhanced_multi",
            "image_size": {"width": image.shape[1], "height": image.shape[0]},
            "parking_status": summary,
            "spaces": [space.to_dict() for space in parking_lot.spaces]
        }
        
        json_path = config.OUTPUT_DIR / f"{image_path.stem}_enhanced_result.json"
        with open(json_path, 'w') as f:
            json.dump(json_output, f, indent=2)
        print(f"JSON output saved to: {json_path}")
    
    return {
        "summary": summary,
        "spaces": parking_lot.spaces
    }

def main():
    parser = argparse.ArgumentParser(description="Enhanced Garage Parking Space Detection")
    parser.add_argument("image", nargs="?", help="Path to image file")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save output files")
    
    args = parser.parse_args()
    
    if not args.image:
        # Process all test images
        test_images = list(config.TEST_IMAGES_DIR.glob("*.jpg")) + \
                     list(config.TEST_IMAGES_DIR.glob("*.png"))
        
        if not test_images:
            print("No images specified and no test images found.")
            sys.exit(1)
        
        print(f"Processing {len(test_images)} test images with enhanced detection...")
        for image_path in test_images:
            process_image(image_path, save_output=not args.no_save)
    else:
        image_path = Path(args.image)
        
        if not image_path.exists():
            print(f"Error: Image file not found: {image_path}")
            sys.exit(1)
        
        process_image(image_path, save_output=not args.no_save)
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)

if __name__ == "__main__":
    main()