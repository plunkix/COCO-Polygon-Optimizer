# COCO Polygon Optimizer

A tool for simplifying and optimizing polygon annotations in COCO format datasets. Useful for reducing complexity in instance segmentation and semantic segmentation annotations.

## Project Background

I developed this tool while working on a seed germination classification project. After using automated annotation tools in CVAT (Computer Vision Annotation Tool) to handle 20-30 seedling instances per image, I encountered significant challenges:

- The automated segmentation tools generated extremely noisy polygons with unnecessary complexity
- The resulting COCO JSON annotations were technically correct but practically unusable
- Each polygon contained excessive vertices that didn't improve segmentation quality
- With 20-30 instances per image, the polygons became unmanageable
- Working with these annotations slowed down model training and inference

Rather than redoing all annotations manually, I developed this optimization tool that:

1. Reduces vertex count by 60-90% while preserving the essential shape
2. Smooths jagged polygon edges for better aesthetics and reduced noise
3. Removes tiny annotation fragments that are likely noise
4. Maintains full COCO format compatibility

## Visualization

[Insert before/after comparison image here showing the difference between original automated annotations and optimized polygons]

## Usage

See [README.frontend.md](README.frontend.md) for instructions on using the Streamlit frontend.
