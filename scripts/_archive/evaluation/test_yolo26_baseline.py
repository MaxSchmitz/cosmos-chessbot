#!/usr/bin/env python3
"""
Test Pretrained YOLO26 Baseline (Before Training)

Tests the pretrained YOLO26 model on chess images to establish a baseline.
This shows what a generic object detector sees vs. our chess-trained model.

Usage:
    python scripts/test_yolo26_baseline.py
    python scripts/test_yolo26_baseline.py --num-samples 20 --visualize
"""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not found")
    print("Install with: pip3 install ultralytics")
    exit(1)


# COCO classes that might be relevant
COCO_RELEVANT_CLASSES = {
    # These are classes from COCO dataset that pretrained YOLO might detect
    0: 'person',
    1: 'bicycle',
    2: 'car',
    39: 'bottle',
    41: 'cup',
    56: 'chair',
    60: 'dining table',
    62: 'laptop',
    63: 'mouse',
    64: 'remote',
    67: 'cell phone',
    73: 'book',
    74: 'clock',
    75: 'vase',
    # Note: COCO doesn't have chess pieces!
}


def test_baseline(
    model_name: str,
    test_dir: Path,
    num_samples: int = 20,
    conf_threshold: float = 0.25,
    visualize: bool = False,
    output_dir: Path = None
):
    """
    Test pretrained YOLO26 on chess images.

    Args:
        model_name: YOLO model to test (e.g., 'yolo26n.pt')
        test_dir: Directory with test images
        num_samples: Number of images to test
        conf_threshold: Confidence threshold
        visualize: Save visualizations
        output_dir: Directory for outputs
    """
    print("=" * 60)
    print("YOLO26 Baseline Test (Pretrained Model)")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Test dir: {test_dir}")
    print(f"Samples: {num_samples}")
    print(f"Confidence threshold: {conf_threshold}")
    print("=" * 60)
    print()

    # Load pretrained YOLO26
    print(f"Loading pretrained model: {model_name}")
    print("(This model is trained on COCO dataset - NOT chess pieces)")
    model = YOLO(model_name)
    print(f"Model loaded: {len(model.names)} classes from COCO dataset")
    print()

    # Get test images
    test_images = sorted(test_dir.glob('chess_*.jpg'))[:num_samples]
    if not test_images:
        print(f"Error: No images found in {test_dir}")
        return

    print(f"Testing on {len(test_images)} images...\n")

    # Create output directory for visualizations
    if visualize and output_dir:
        vis_dir = output_dir / 'baseline_visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)

    # Statistics
    detection_stats = defaultdict(int)
    total_detections = 0
    images_with_detections = 0

    results_summary = []

    for i, image_path in enumerate(test_images):
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not load {image_path}")
            continue

        # Run detection
        results = model(image, conf=conf_threshold, verbose=False)

        # Parse results
        detections = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            images_with_detections += 1

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                class_name = model.names[cls_id]
                detection_stats[class_name] += 1
                total_detections += 1

                detections.append({
                    'class': class_name,
                    'class_id': cls_id,
                    'confidence': conf,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })

        # Log progress
        print(f"[{i+1}/{len(test_images)}] {image_path.name}: "
              f"{len(detections)} detections")

        if detections:
            for det in detections:
                print(f"  - {det['class']} ({det['confidence']:.2f})")

        results_summary.append({
            'image': image_path.name,
            'num_detections': len(detections),
            'detections': detections
        })

        # Visualize
        if visualize and output_dir:
            vis_image = image.copy()

            # Draw detections
            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class']} {det['confidence']:.2f}"
                cv2.putText(vis_image, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save
            vis_path = vis_dir / image_path.name
            cv2.imwrite(str(vis_path), vis_image)

    # Print summary
    print("\n" + "=" * 60)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 60)
    print(f"Images tested: {len(test_images)}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Total detections: {total_detections}")
    print(f"Avg detections per image: {total_detections / len(test_images):.2f}")
    print()

    if detection_stats:
        print("Detected classes (from COCO dataset):")
        for class_name, count in sorted(detection_stats.items(), key=lambda x: -x[1]):
            print(f"  {class_name}: {count}")
    else:
        print("No detections found!")
        print("This is expected - pretrained YOLO26 doesn't know about chess pieces.")

    print()
    print("Note: YOLO26 is pretrained on COCO dataset which contains:")
    print("  - 80 common objects (person, car, dog, etc.)")
    print("  - NO chess pieces!")
    print()
    print("After training on our chess dataset, we expect:")
    print("  - ~40-60 detections per image (one per piece)")
    print("  - 12 chess piece classes (white/black × 6 piece types)")
    print("  - >95% mAP50 accuracy")
    print("=" * 60)

    # Save results
    if output_dir:
        results_file = output_dir / 'baseline_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'model': model_name,
                'num_images': len(test_images),
                'images_with_detections': images_with_detections,
                'total_detections': total_detections,
                'detection_stats': dict(detection_stats),
                'per_image_results': results_summary
            }, f, indent=2)
        print(f"\nResults saved to {results_file}")

    return {
        'total_detections': total_detections,
        'images_with_detections': images_with_detections,
        'detection_stats': dict(detection_stats)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test pretrained YOLO26 baseline on chess images"
    )

    parser.add_argument(
        '--model',
        type=str,
        default='yolo26n.pt',
        help='Pretrained YOLO26 model (yolo26n.pt, yolo26s.pt, yolo26m.pt, etc.)'
    )
    parser.add_argument(
        '--test-dir',
        type=Path,
        default=Path('data/yolo26_chess/images/val'),
        help='Directory with test images'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=20,
        help='Number of images to test (default: 20)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Save visualization images'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('baseline_results'),
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Validate test directory
    if not args.test_dir.exists():
        print(f"Error: Test directory not found: {args.test_dir}")
        print("Have you converted the dataset to YOLO format?")
        print("Run: python3 scripts/convert_llava_to_yolo26_v2.py")
        exit(1)

    # Create output directory
    if args.visualize or True:  # Always create for results.json
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run baseline test
    test_baseline(
        model_name=args.model,
        test_dir=args.test_dir,
        num_samples=args.num_samples,
        conf_threshold=args.conf,
        visualize=args.visualize,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
