#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO Polygon Simplification Tool - WSL Compatible OpenCV-only Version

This script optimizes COCO format annotations by:
1. Simplifying polygons to reduce vertex count
2. Smoothing polygon curves
3. Removing tiny polygons (likely noise)

This version uses only OpenCV for visualization, avoiding matplotlib/Qt issues in WSL.
"""

import argparse
import json
import os
import numpy as np
import cv2
from shapely.geometry import Polygon
from tqdm import tqdm
import random

def simplify_polygon(polygon_points, epsilon=1.0):
    """
    Simplify a polygon using Douglas-Peucker algorithm
    
    Args:
        polygon_points: numpy array of points with shape (N, 2)
        epsilon: simplification threshold - higher values create simpler polygons
        
    Returns:
        Simplified polygon as numpy array
    """
    # Check if polygon has at least 3 points
    if polygon_points.shape[0] < 3:
        return polygon_points
        
    # Convert to the format expected by cv2.approxPolyDP
    contour = polygon_points.reshape(-1, 1, 2).astype(np.float32)
    
    # Apply Douglas-Peucker algorithm
    simplified = cv2.approxPolyDP(contour, epsilon, True)
    
    # Convert back to the original format
    return simplified.reshape(-1, 2)
    
def simplify_ramer_douglas_peucker(points, epsilon=1.0):
    """
    Alternative implementation of Ramer-Douglas-Peucker algorithm
    Sometimes produces better results for natural shapes like seedlings
    
    Args:
        points: numpy array of points with shape (N, 2)
        epsilon: simplification threshold
        
    Returns:
        Simplified polygon as numpy array
    """
    if len(points) < 3:
        return points
        
    def distance_point_to_line(point, line_start, line_end):
        """Calculate perpendicular distance from point to line"""
        if np.array_equal(line_start, line_end):
            return np.linalg.norm(point - line_start)
            
        n = np.linalg.norm(line_end - line_start)
        if n == 0:
            return np.linalg.norm(point - line_start)
            
        return np.linalg.norm(np.cross(line_end - line_start, line_start - point)) / n
    
    def rdp(points, epsilon, start_idx, end_idx):
        """Recursive Douglas-Peucker implementation"""
        if end_idx - start_idx <= 1:
            return []
            
        # Find the point with the maximum distance
        dmax = 0
        index = start_idx
        for i in range(start_idx + 1, end_idx):
            d = distance_point_to_line(points[i], points[start_idx], points[end_idx])
            if d > dmax:
                index = i
                dmax = d
                
        # If max distance is greater than epsilon, recursively simplify
        result = []
        if dmax > epsilon:
            rec1 = rdp(points, epsilon, start_idx, index)
            rec2 = rdp(points, epsilon, index, end_idx)
            result = rec1 + [index] + rec2
        
        return result
    
    # Run the algorithm
    indices = [0] + rdp(points, epsilon, 0, len(points) - 1) + [len(points) - 1]
    indices = sorted(set(indices))  # Remove duplicates and sort
    
    return points[indices]

def smooth_polygon(polygon_points, smoothing_factor=0.2, iterations=1):
    """
    Apply smoothing to a polygon
    
    Args:
        polygon_points: numpy array of points with shape (N, 2)
        smoothing_factor: factor to control smoothing amount (0-1)
        iterations: number of smoothing passes to apply
        
    Returns:
        Smoothed polygon as numpy array
    """
    if len(polygon_points) <= 3:
        return polygon_points
    
    # Apply multiple iterations of smoothing if requested
    smoothed = polygon_points.copy()
    for _ in range(iterations):
        # Create a temporary array for this iteration
        temp_smoothed = np.zeros_like(smoothed)
        
        # Apply smoothing (Chaikin's algorithm variant)
        for i in range(len(smoothed)):
            prev_idx = (i - 1) % len(smoothed)
            next_idx = (i + 1) % len(smoothed)
            
            # Weight current point more than neighbors
            temp_smoothed[i] = (1 - smoothing_factor) * smoothed[i] + \
                          (smoothing_factor / 2) * smoothed[prev_idx] + \
                          (smoothing_factor / 2) * smoothed[next_idx]
        
        smoothed = temp_smoothed
    
    return smoothed

def chaikin_smooth(polygon_points, iterations=2):
    """
    Apply Chaikin's smoothing algorithm to a polygon
    Great for natural-looking curves while preserving overall shape
    
    Args:
        polygon_points: numpy array of points with shape (N, 2)
        iterations: number of subdivision iterations
        
    Returns:
        Smoothed polygon as numpy array
    """
    if len(polygon_points) < 3:
        return polygon_points
    
    # Make sure the polygon is closed
    if not np.array_equal(polygon_points[0], polygon_points[-1]):
        points = np.vstack([polygon_points, polygon_points[0]])
    else:
        points = polygon_points.copy()
    
    # Apply Chaikin's algorithm iteratively
    for _ in range(iterations):
        new_points = []
        
        # Process each edge
        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i+1]
            
            # Calculate 1/4 and 3/4 points along the edge
            q0 = p0 * 0.75 + p1 * 0.25
            q1 = p0 * 0.25 + p1 * 0.75
            
            new_points.extend([q0, q1])
            
        # Close the polygon
        points = np.array(new_points)
        if not np.array_equal(points[0], points[-1]):
            points = np.vstack([points, points[0]])
    
    # Remove the duplicate closing point
    return points[:-1]

def create_example_visualization(output_path):
    """
    Create a sample comparison image using OpenCV (no matplotlib dependency)
    """
    # Create a beige background to simulate seedling tray
    width, height = 1600, 800
    background = np.ones((height, width, 3), dtype=np.uint8) * np.array([160, 200, 220], dtype=np.uint8)  # BGR for OpenCV
    
    # Create a side-by-side image
    original = background.copy()
    optimized = background.copy()
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(original, "Original Annotations", (50, 50), font, 1.5, (0, 0, 0), 3)
    cv2.putText(optimized, "Optimized Annotations", (50, 50), font, 1.5, (0, 0, 0), 3)
    
    # Create some seedling shapes
    centers = [(200, 200), (400, 300), (600, 250), (300, 500), (500, 450)]
    radius = 50
    
    total_original_pts = 0
    total_optimized_pts = 0
    
    for center_x, center_y in centers:
        # Complex polygon (original)
        complex_points = []
        num_points = 100  # Lots of points for the original
        
        for i in range(num_points):
            angle = i * (2 * np.pi / num_points)
            # Add some noise to the radius
            r = radius + np.random.normal(0, 10)
            x = int(center_x + r * np.cos(angle))
            y = int(center_y + r * np.sin(angle))
            complex_points.append([x, y])
            
        complex_points = np.array(complex_points, dtype=np.int32)
        total_original_pts += len(complex_points)
        
        # Simple polygon (optimized)
        simple_points = []
        num_simple_points = 10  # Fewer points for the simplified version
        
        for i in range(num_simple_points):
            angle = i * (2 * np.pi / num_simple_points)
            # Similar shape but fewer points
            r = radius + np.random.normal(0, 5)
            x = int(center_x + r * np.cos(angle))
            y = int(center_y + r * np.sin(angle))
            simple_points.append([x, y])
            
        simple_points = np.array(simple_points, dtype=np.int32)
        total_optimized_pts += len(simple_points)
        
        # Draw polygons - lime green is (0, 255, 0) in BGR
        cv2.polylines(original, [complex_points], True, (0, 255, 0), 2)
        cv2.polylines(optimized, [simple_points], True, (0, 255, 0), 2)
    
    # Add point count
    cv2.putText(original, f"Total points: {total_original_pts}", (50, height - 50), font, 1, (0, 0, 0), 2)
    cv2.putText(optimized, f"Total points: {total_optimized_pts}", (50, height - 50), font, 1, (0, 0, 0), 2)
    
    # Calculate reduction percentage
    reduction = ((total_original_pts - total_optimized_pts) / total_original_pts) * 100
    
    # Combine images side by side
    combined = np.hstack((original, optimized))
    
    # Add title
    title_bar = np.ones((100, combined.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(title_bar, f"Polygon Simplification - Point Reduction: {reduction:.1f}%", 
                (50, 60), font, 1.5, (0, 0, 0), 3)
    
    # Combine with title
    final_image = np.vstack((title_bar, combined))
    
    # Save image
    cv2.imwrite(output_path, final_image)
    print(f"Created example visualization: {output_path}")

def optimize_coco_annotations(input_file, output_file, epsilon=2.0, min_area=25, smooth=True, max_points=50, min_points=5):
    """
    Optimize COCO annotation file by simplifying polygons
    
    Args:
        input_file: Path to input COCO JSON file
        output_file: Path to output optimized COCO JSON file
        epsilon: Douglas-Peucker simplification parameter
        min_area: Minimum polygon area to keep (small polygons are often noise)
        smooth: Whether to apply polygon smoothing
        max_points: Maximum number of points to allow per polygon
        min_points: Minimum number of points to require per polygon
    """
    print(f"Loading COCO annotations from {input_file}...")
    with open(input_file, 'r') as f:
        coco_data = json.load(f)
    
    print(f"Found {len(coco_data['annotations'])} annotations")
    
    # Process each annotation
    optimized_annotations = []
    removed_count = 0
    simplified_points_count = 0
    original_points_count = 0
    
    for ann in tqdm(coco_data['annotations'], desc="Optimizing polygons"):
        if 'segmentation' not in ann or not ann['segmentation']:
            # Skip annotations without segmentation
            optimized_annotations.append(ann)
            continue
        
        # Get the first polygon (COCO can have multiple polygons per annotation)
        if isinstance(ann['segmentation'], list) and len(ann['segmentation']) > 0:
            all_optimized_segments = []
            
            for seg in ann['segmentation']:
                if len(seg) < 6:  # Skip invalid polygons (need at least 3 points)
                    continue
                
                # Reshape to x,y points
                points = np.array(seg).reshape(-1, 2)
                original_points_count += len(points)
                
                # Calculate polygon area
                try:
                    polygon = Polygon(points)
                    if not polygon.is_valid or polygon.area < min_area:
                        removed_count += 1
                        continue
                except Exception as e:
                    # Skip invalid polygons that can't be processed by Shapely
                    continue
                
                # Simplify the polygon using adaptive epsilon to target max_points
                simplified = simplify_polygon(points, epsilon)
                
                # If still too many points, increase epsilon adaptively until within max_points
                current_epsilon = epsilon
                while len(simplified) > max_points and current_epsilon < 50:
                    current_epsilon *= 1.5
                    simplified = simplify_polygon(points, current_epsilon)
                
                # If too few points, decrease epsilon to maintain minimum detail
                current_epsilon = epsilon
                while len(simplified) < min_points and current_epsilon > 0.5:
                    current_epsilon *= 0.7
                    simplified = simplify_polygon(points, current_epsilon)
                
                # Try using alternative RDP algorithm if we have a complex shape
                if len(simplified) > max_points:
                    simplified_alt = simplify_ramer_douglas_peucker(points, current_epsilon*1.2)
                    # Use whichever method gives fewer points while keeping above min_points
                    if len(simplified_alt) < len(simplified) and len(simplified_alt) >= min_points:
                        simplified = simplified_alt
                
                # Optional: smooth the polygon using Chaikin for better plant shapes
                if smooth and len(simplified) > 3:
                    if len(simplified) > 20:  # For very detailed polygons
                        simplified = chaikin_smooth(simplified, iterations=2)
                    else:
                        simplified = smooth_polygon(simplified, smoothing_factor=0.2, iterations=2)
                
                # Skip invalid polygons after simplification
                if len(simplified) < 3:
                    continue
                    
                # Force polygon to be closed
                if not np.array_equal(simplified[0], simplified[-1]):
                    simplified = np.vstack([simplified, simplified[0]])
                    
                simplified_points_count += len(simplified)
                
                # Convert back to COCO format
                optimized_segment = simplified.flatten().tolist()
                all_optimized_segments.append(optimized_segment)
            
            if all_optimized_segments:
                ann['segmentation'] = all_optimized_segments
                optimized_annotations.append(ann)
            else:
                removed_count += 1
        else:
            optimized_annotations.append(ann)
    
    # Update annotations in the COCO data
    coco_data['annotations'] = optimized_annotations
    
    # Save optimized file
    with open(output_file, 'w') as f:
        json.dump(coco_data, f)
    
    reduction = (1 - (simplified_points_count / original_points_count)) * 100 if original_points_count > 0 else 0
    print(f"Optimization complete!")
    print(f"Original annotations: {len(coco_data['annotations']) + removed_count}")
    print(f"Optimized annotations: {len(optimized_annotations)}")
    print(f"Removed annotations: {removed_count}")
    print(f"Original points: {original_points_count}")
    print(f"Simplified points: {simplified_points_count}")
    print(f"Point reduction: {reduction:.2f}%")
    print(f"Optimized file saved to: {output_file}")
    
    # Return statistics
    return {
        "original_annotations": len(coco_data['annotations']) + removed_count,
        "optimized_annotations": len(optimized_annotations),
        "removed_annotations": removed_count,
        "original_points": original_points_count,
        "simplified_points": simplified_points_count,
        "reduction": reduction
    }

def generate_comparisons_opencv(
    original_file, 
    optimized_file, 
    image_folder, 
    output_folder,
    num_samples=5
):
    """
    Generate comparison visualizations using OpenCV (no matplotlib dependency)
    
    Args:
        original_file: Path to original COCO JSON file
        optimized_file: Path to optimized COCO JSON file
        image_folder: Path to folder containing images
        output_folder: Path to save visualization results
        num_samples: Number of sample images to visualize
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load original and optimized data
    with open(original_file, "r") as f:
        original_data = json.load(f)
    
    with open(optimized_file, "r") as f:
        optimized_data = json.load(f)
    
    # Create image id to annotations mapping
    original_anns_by_image = {}
    for ann in original_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in original_anns_by_image:
            original_anns_by_image[img_id] = []
        original_anns_by_image[img_id].append(ann)
    
    optimized_anns_by_image = {}
    for ann in optimized_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in optimized_anns_by_image:
            optimized_anns_by_image[img_id] = []
        optimized_anns_by_image[img_id].append(ann)
    
    # Get image info mapping
    image_info = {img["id"]: img for img in original_data["images"]}
    
    # Select random sample of images with annotations in both sets
    common_img_ids = list(set(original_anns_by_image.keys()) & set(optimized_anns_by_image.keys()))
    if not common_img_ids:
        print("No common images found with annotations in both sets!")
        # Create an example visualization instead
        create_example_visualization(os.path.join(output_folder, "example_comparison.png"))
        return
    
    sample_img_ids = random.sample(common_img_ids, min(num_samples, len(common_img_ids)))
    
    # Try all possible image paths
    images_found = 0
    
    for img_id in sample_img_ids:
        if img_id not in image_info:
            print(f"Warning: Image ID {img_id} not found in image info")
            continue
            
        img_info = image_info[img_id]
        if "file_name" not in img_info:
            print(f"Warning: file_name not found for image ID {img_id}")
            continue
        
        # Handle the case where file_name might be None
        file_name = img_info.get("file_name", "")
        if not file_name:
            print(f"Warning: Empty file_name for image ID {img_id}")
            continue
        
        # Try multiple possible paths
        possible_paths = []
        try:
            # Path from file_name as is
            possible_paths.append(os.path.join(image_folder, file_name))
            # Just the basename
            possible_paths.append(os.path.join(image_folder, os.path.basename(file_name)))
            # Lowercase extension
            if '.' in file_name:
                base, ext = os.path.splitext(file_name)
                possible_paths.append(os.path.join(image_folder, base + ext.lower()))
                possible_paths.append(os.path.join(image_folder, os.path.basename(base + ext.lower())))
                # Uppercase extension
                possible_paths.append(os.path.join(image_folder, base + ext.upper()))
                possible_paths.append(os.path.join(image_folder, os.path.basename(base + ext.upper())))
            # Add path with .JPG if filename doesn't have extension
            else:
                possible_paths.append(os.path.join(image_folder, f"{file_name}.JPG"))
                possible_paths.append(os.path.join(image_folder, f"{file_name}.jpg"))
        except Exception as e:
            print(f"Error creating paths for {file_name}: {e}")
            continue
        
        # Check if any of these paths exist
        image_path = None
        for path in possible_paths:
            if os.path.exists(path):
                image_path = path
                break
                
        if image_path is None:
            print(f"Could not find image for ID {img_id}. Tried:")
            for path in possible_paths:
                print(f"  - {path}")
            continue
            
        # Found an image!
        print(f"Found image: {image_path}")
            
        # Load image
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
        
        # Create copies for original and optimized annotations
        original_img = image.copy()
        optimized_img = image.copy()
        
        # Draw original annotations
        total_original_points = 0
        for ann in original_anns_by_image[img_id]:
            if "segmentation" in ann and isinstance(ann["segmentation"], list):
                for seg in ann["segmentation"]:
                    try:
                        points = np.array(seg).reshape(-1, 2).astype(np.int32)
                        total_original_points += len(points)
                        cv2.polylines(original_img, [points], True, (0, 255, 0), 2)  # Green color
                    except Exception as e:
                        print(f"Error drawing original polygon: {e}")
                        continue
        
        # Draw optimized annotations
        total_optimized_points = 0
        for ann in optimized_anns_by_image[img_id]:
            if "segmentation" in ann and isinstance(ann["segmentation"], list):
                for seg in ann["segmentation"]:
                    try:
                        points = np.array(seg).reshape(-1, 2).astype(np.int32)
                        total_optimized_points += len(points)
                        cv2.polylines(optimized_img, [points], True, (0, 255, 0), 2)  # Green color
                    except Exception as e:
                        print(f"Error drawing optimized polygon: {e}")
                        continue
        
        # Calculate reduction
        reduction = ((total_original_points - total_optimized_points) / total_original_points) * 100 if total_original_points > 0 else 0
        
        # Add text with statistics
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(original_img, f"Original Annotations: {len(original_anns_by_image[img_id])}", 
                   (10, 30), font, 1, (0, 0, 0), 2)
        cv2.putText(original_img, f"Total points: {total_original_points}", 
                   (10, 70), font, 1, (0, 0, 0), 2)
                   
        cv2.putText(optimized_img, f"Optimized Annotations: {len(optimized_anns_by_image[img_id])}", 
                   (10, 30), font, 1, (0, 0, 0), 2)
        cv2.putText(optimized_img, f"Total points: {total_optimized_points}", 
                   (10, 70), font, 1, (0, 0, 0), 2)
        
        # Create a title bar with the reduction percentage
        h, w = original_img.shape[:2]
        title_height = 100
        title_bar = np.ones((title_height, w*2, 3), dtype=np.uint8) * 255  # White background
        cv2.putText(title_bar, f"Polygon Simplification - Point Reduction: {reduction:.1f}%", 
                   (10, 60), font, 1.5, (0, 0, 0), 3)
        
        # Combine images side by side with title
        if original_img.shape[0] != optimized_img.shape[0] or original_img.shape[1] != optimized_img.shape[1]:
            # Resize if dimensions don't match
            optimized_img = cv2.resize(optimized_img, (original_img.shape[1], original_img.shape[0]))
            
        combined = np.hstack((original_img, optimized_img))
        final_image = np.vstack((title_bar, combined))
        
        # Save result
        output_path = os.path.join(output_folder, f"comparison_{img_id}.png")
        cv2.imwrite(output_path, final_image)
        print(f"Saved comparison for image {img_id} to {output_path}")
        images_found += 1
        
    if images_found == 0:
        print("\nWARNING: Could not find any images matching the IDs in the COCO file.")
        print("Creating an example visualization instead...")
        create_example_visualization(os.path.join(output_folder, "example_comparison.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize COCO annotations by simplifying polygons")
    parser.add_argument("--input", required=True, help="Input COCO JSON file")
    parser.add_argument("--output", required=True, help="Output optimized COCO JSON file")
    parser.add_argument("--epsilon", type=float, default=2.0, help="Douglas-Peucker simplification parameter")
    parser.add_argument("--min-area", type=float, default=25.0, help="Minimum polygon area to keep")
    parser.add_argument("--no-smooth", action="store_true", help="Disable polygon smoothing")
    parser.add_argument("--max-points", type=int, default=50, help="Maximum points per polygon")
    parser.add_argument("--min-points", type=int, default=5, help="Minimum points per polygon")
    parser.add_argument("--visualize", action="store_true", help="Generate comparison visualizations")
    parser.add_argument("--image-folder", help="Path to image folder (required for visualization)")
    parser.add_argument("--vis-output", default="comparison_results", help="Output folder for visualizations")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to visualize")
    
    args = parser.parse_args()
    
    # Run optimization
    optimize_coco_annotations(
        args.input, 
        args.output, 
        epsilon=args.epsilon, 
        min_area=args.min_area, 
        smooth=not args.no_smooth,
        max_points=args.max_points,
        min_points=args.min_points
    )
    
    # Generate visualizations if requested
    if args.visualize:
        if not args.image_folder:
            print("Error: --image-folder is required for visualization")
        else:
            generate_comparisons_opencv(
                args.input,
                args.output,
                args.image_folder,
                args.vis_output,
                args.num_samples
            )