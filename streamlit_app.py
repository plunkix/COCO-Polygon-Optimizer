import streamlit as st
import os
import json
import tempfile
import numpy as np
import cv2
from PIL import Image
import io
import sys
import subprocess
import base64

# Page configuration
st.set_page_config(
    page_title="COCO Polygon Optimizer", 
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure the optimizer module is in the path
sys.path.append('/app')

# Import the optimizer functions (modify this if needed)
try:
    from coco_polygon_optimizer import optimize_coco_annotations, generate_comparisons_opencv
except ImportError:
    # For testing or development outside the Docker container
    st.warning("Could not import optimizer module directly. Using subprocess method.")
    
    def optimize_coco_annotations(input_file, output_file, **kwargs):
        """Fallback function that calls the optimizer script as a subprocess"""
        cmd = ["python", "coco_polygon_optimizer.py", 
               "--input", input_file, 
               "--output", output_file]
        
        # Add command line arguments based on kwargs
        if 'epsilon' in kwargs:
            cmd.extend(["--epsilon", str(kwargs['epsilon'])])
        if 'min_area' in kwargs:
            cmd.extend(["--min-area", str(kwargs['min_area'])])
        if 'max_points' in kwargs:
            cmd.extend(["--max-points", str(kwargs['max_points'])])
        if 'min_points' in kwargs:
            cmd.extend(["--min-points", str(kwargs['min_points'])])
        if 'smooth' in kwargs and not kwargs['smooth']:
            cmd.extend(["--no-smooth"])
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse output to extract statistics
        stats = {
            'original_annotations': 0,
            'optimized_annotations': 0,
            'removed_annotations': 0,
            'original_points': 0,
            'simplified_points': 0,
            'reduction': 0.0
        }
        
        for line in result.stdout.split('\n'):
            if "Original annotations:" in line:
                stats['original_annotations'] = int(line.split(':')[1].strip())
            elif "Optimized annotations:" in line:
                stats['optimized_annotations'] = int(line.split(':')[1].strip())
            elif "Removed annotations:" in line:
                stats['removed_annotations'] = int(line.split(':')[1].strip())
            elif "Original points:" in line:
                stats['original_points'] = int(line.split(':')[1].strip())
            elif "Simplified points:" in line:
                stats['simplified_points'] = int(line.split(':')[1].strip())
            elif "Point reduction:" in line:
                stats['reduction'] = float(line.split(':')[1].strip().rstrip('%'))
        
        return stats
    
    def generate_comparisons_opencv(input_file, output_file, image_folder, output_folder, num_samples=5):
        """Fallback function that calls the optimizer script for visualization"""
        cmd = ["python", "coco_polygon_optimizer.py", 
               "--input", input_file, 
               "--output", output_file,
               "--visualize",
               "--image-folder", image_folder,
               "--vis-output", output_folder,
               "--num-samples", str(num_samples)]
        
        subprocess.run(cmd, capture_output=True, text=True)



# Custom CSS
st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton button {
        width: 100%;
        height: 3rem;
        font-weight: bold;
        font-size: 1.1rem;
        margin-top: 1rem;
    }
    .upload-title {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for project information
with st.sidebar:
    st.title("COCO Polygon Optimizer")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
        This tool simplifies and optimizes polygon annotations in COCO format datasets. 
        Useful for reducing complexity in instance segmentation and semantic segmentation annotations.
    """)
    
    st.markdown("### Features")
    st.markdown("""
        - **Polygon Simplification**: Reduce vertices while preserving shape
        - **Polygon Smoothing**: Create more natural contours
        - **Small Polygon Removal**: Eliminate tiny annotation fragments
        - **Visualization**: Compare original vs. optimized annotations
    """)
    
    st.markdown("### GitHub")
    st.markdown("[View on GitHub](https://github.com/plunkix/COCO-Polygon-Optimizer)")
    
    st.markdown("---")
    st.markdown("© 2025 plunkix")

# Main content
st.title("COCO Polygon Annotation Optimizer")
st.markdown("Upload your COCO annotation file and configure optimization parameters to simplify polygon masks.")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Upload & Configure", "Results", "Documentation"])

with tab1:
    # File upload section
    st.header("1. Upload Files")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="upload-title">COCO JSON Annotations</p>', unsafe_allow_html=True)
        coco_file = st.file_uploader("", type=["json"])
        if coco_file is not None:
            st.success(f"Uploaded: {coco_file.name}")
            
            # Preview JSON structure
            try:
                coco_data = json.load(coco_file)
                coco_file.seek(0)  # Reset file pointer after reading
                
                st.markdown("**Dataset Stats:**")
                ann_count = len(coco_data.get("annotations", []))
                img_count = len(coco_data.get("images", []))
                cat_count = len(coco_data.get("categories", []))
                
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                with stats_col1:
                    st.metric("Annotations", ann_count)
                with stats_col2:
                    st.metric("Images", img_count)
                with stats_col3:
                    st.metric("Categories", cat_count)
            except:
                st.warning("Could not parse JSON file. Please check the format.")
                
    with col2:
        st.markdown('<p class="upload-title">Images (Optional, for Visualization)</p>', unsafe_allow_html=True)
        image_files = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if image_files:
            st.success(f"Uploaded {len(image_files)} images")
            
            # Display image thumbnails
            if len(image_files) > 0:
                image_cols = st.columns(min(3, len(image_files)))
                for i, img_col in enumerate(image_cols):
                    if i < len(image_files):
                        try:
                            img = Image.open(image_files[i])
                            img_col.image(img, caption=image_files[i].name, width=150)
                        except:
                            img_col.warning(f"Could not preview {image_files[i].name}")

    # Parameters section
    st.header("2. Configure Optimization")
    
    with st.expander("Simplification Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            epsilon = st.slider(
                "Simplification Epsilon", 
                min_value=0.5, 
                max_value=10.0, 
                value=2.0, 
                step=0.5,
                help="Controls the level of simplification. Higher values result in fewer points."
            )
            
            min_area = st.slider(
                "Minimum Area (px²)", 
                min_value=0.0, 
                max_value=100.0, 
                value=25.0, 
                step=1.0,
                help="Polygons smaller than this area will be removed (likely noise)."
            )
            
        with col2:
            max_points = st.slider(
                "Maximum Points per Polygon", 
                min_value=5, 
                max_value=100, 
                value=50, 
                step=5,
                help="Target maximum number of points per polygon."
            )
            
            min_points = st.slider(
                "Minimum Points per Polygon", 
                min_value=3, 
                max_value=20, 
                value=5, 
                step=1,
                help="Ensure polygons maintain at least this many points."
            )
    
    with st.expander("Smoothing Options", expanded=True):
        smooth_enabled = st.checkbox("Enable Smoothing", value=True)
        
        if smooth_enabled:
            col1, col2 = st.columns(2)
            with col1:
                smooth_method = st.selectbox(
                    "Smoothing Algorithm", 
                    options=["moving-avg", "chaikin"],
                    index=1,
                    help="Chaikin typically produces better natural curves."
                )
            with col2:
                smooth_iterations = st.slider(
                    "Smoothing Iterations", 
                    min_value=1, 
                    max_value=5, 
                    value=2, 
                    step=1,
                    help="More iterations = smoother curves, but may lose detail."
                )
        
    # Visualization options
    with st.expander("Visualization Options", expanded=True):
        visualize_enabled = st.checkbox("Generate Comparison Visualizations", value=True)
        
        if visualize_enabled:
            if not image_files:
                st.warning("Upload images to enable visualization")
                
            num_samples = st.slider(
                "Number of Sample Images", 
                min_value=1, 
                max_value=10, 
                value=3, 
                step=1,
                help="Number of random images to generate comparisons for."
            )
            
    # Process button
    if st.button("Optimize COCO Annotations", use_container_width=True):
        if coco_file is None:
            st.error("Please upload a COCO JSON file")
        else:
            # Move to results tab
            tab2.active = True
            
            # Create temp directories
            with st.spinner("Setting up processing environment..."):
                temp_dir = tempfile.TemporaryDirectory()
                input_dir = os.path.join(temp_dir.name, "input")
                output_dir = os.path.join(temp_dir.name, "output") 
                images_dir = os.path.join(temp_dir.name, "images")
                vis_dir = os.path.join(output_dir, "visualizations")
                
                os.makedirs(input_dir, exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(vis_dir, exist_ok=True)
            
            # Save input files to temp directory
            with st.spinner("Saving input files..."):
                input_path = os.path.join(input_dir, "annotations.json")
                with open(input_path, "wb") as f:
                    f.write(coco_file.getbuffer())
                
                # Save images if provided
                if image_files and visualize_enabled:
                    for img_file in image_files:
                        img_path = os.path.join(images_dir, img_file.name)
                        with open(img_path, "wb") as f:
                            f.write(img_file.getbuffer())
            
            # Run optimization
            with st.spinner("Optimizing annotations... This may take a few minutes."):
                output_path = os.path.join(output_dir, "optimized.json")
                
                # Run optimization
                try:
                    stats = optimize_coco_annotations(
                        input_path, 
                        output_path, 
                        epsilon=epsilon,
                        min_area=min_area,
                        smooth=smooth_enabled,
                        max_points=max_points,
                        min_points=min_points
                    )
                    
                    # Generate visualizations if requested
                    if visualize_enabled and image_files:
                        generate_comparisons_opencv(
                            input_path,
                            output_path,
                            images_dir,
                            vis_dir,
                            num_samples=num_samples
                        )
                    
                    # Store results in session state for the results tab
                    st.session_state.optimization_complete = True
                    st.session_state.stats = stats
                    st.session_state.output_path = output_path
                    st.session_state.vis_dir = vis_dir
                    
                    # Switch to results tab
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"Optimization failed: {str(e)}")
                    st.exception(e)

with tab2:
    if not hasattr(st.session_state, 'optimization_complete') or not st.session_state.optimization_complete:
        st.info("Run optimization to see results here")
    else:
        stats = st.session_state.stats
        output_path = st.session_state.output_path
        vis_dir = st.session_state.vis_dir
        
        # Success message
        st.success(f"✅ Optimization complete! Point reduction: {stats['reduction']:.2f}%")
        
        # Statistics section
        st.header("Optimization Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Point Reduction", f"{stats['reduction']:.2f}%")
        with col2:
            st.metric("Original Points", stats['original_points'])
        with col3:
            st.metric("Optimized Points", stats['simplified_points'])
            
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Annotations", stats['original_annotations'])
        with col2:
            st.metric("Optimized Annotations", stats['optimized_annotations'])
        with col3:
            st.metric("Removed Annotations", stats['removed_annotations'])
        
        # Download results
        st.header("Download Results")
        
        if os.path.exists(output_path):
            with open(output_path, "rb") as f:
                optimized_data = f.read()
                
                file_size = len(optimized_data) / 1024  # KB
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.download_button(
                        "Download Optimized COCO JSON",
                        optimized_data,
                        "optimized_annotations.json",
                        "application/json",
                        use_container_width=True
                    )
                with col2:
                    st.info(f"{file_size:.1f} KB")
        
        # Visualization results
        if os.path.exists(vis_dir):
            st.header("Visualizations")
            
            vis_files = [f for f in os.listdir(vis_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
            
            if vis_files:
                for vis_file in vis_files:
                    vis_path = os.path.join(vis_dir, vis_file)
                    st.image(vis_path, caption=f"Comparison: {vis_file}", use_column_width=True)
                    
                    # Add download button for each visualization
                    with open(vis_path, "rb") as f:
                        img_bytes = f.read()
                        st.download_button(
                            f"Download {vis_file}",
                            img_bytes,
                            vis_file,
                            "image/png",
                        )
                    
                    st.markdown("---")
            else:
                st.info("No visualizations were generated. Check that images were provided and visualization was enabled.")

with tab3:
    st.header("Documentation")
    
    st.subheader("About COCO Polygon Optimizer")
    st.markdown("""
        The COCO Polygon Optimizer is a tool for simplifying and optimizing polygon annotations in 
        COCO (Common Objects in Context) format datasets. It's particularly useful for reducing complexity 
        in instance segmentation and semantic segmentation annotations.
    """)
    
    st.subheader("Key Features")
    st.markdown("""
    - **Polygon Simplification**: Reduce the number of vertices in polygon masks while preserving shape
    - **Polygon Smoothing**: Create more natural contours for improved aesthetics and reduced noise
    - **Small Polygon Removal**: Eliminate tiny annotation fragments that are likely noise
    - **Visualization Tools**: Generate comparison images to validate optimizations
    - **Batch Processing**: Process entire datasets with a single command
    - **Format Preservation**: Maintains full COCO format compatibility
    """)
    
    st.subheader("Parameters Guide")
    
    st.markdown("**Simplification Parameters**")
    st.markdown("""
    - **Epsilon**: Controls the level of simplification in the Douglas-Peucker algorithm. Higher values result in more aggressive simplification.
    - **Minimum Area**: Polygons smaller than this area (in square pixels) will be removed as they're likely noise.
    - **Maximum Points**: Target maximum number of points to keep per polygon.
    - **Minimum Points**: Ensures polygons maintain at least this many points to preserve basic shape.
    """)
    
    st.markdown("**Smoothing Options**")
    st.markdown("""
    - **Smoothing Algorithm**:
        - **Moving Average**: Simple smoothing that averages neighboring points.
        - **Chaikin**: Creates smoother curves by recursively replacing each edge with two edges.
    - **Smoothing Iterations**: Number of times to apply the smoothing algorithm. More iterations create smoother results but may lose detail.
    """)
    
    st.markdown("**Use Cases**")
    st.markdown("""
    - **Improved Model Training**: Reduce overfitting to unnecessarily complex boundaries
    - **Faster Inference**: Simplified polygons require less computation for rendering
    - **Reduced File Size**: Smaller annotation files for easier storage and sharing
    - **Better Visualization**: Cleaner, more interpretable polygon overlays
    """)
    
    st.subheader("COCO Format Reference")
    with st.expander("COCO Annotation Format Example"):
        st.code("""
{
  "info": {...},
  "licenses": [...],
  "categories": [
    {"id": 1, "name": "person", "supercategory": "person"},
    ...
  ],
  "images": [
    {
      "id": 1,
      "file_name": "000000001.jpg",
      "width": 640,
      "height": 480,
      ...
    },
    ...
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "area": 12345.0,
      "bbox": [x, y, width, height],
      "iscrowd": 0
    },
    ...
  ]
}
        """, language="json")