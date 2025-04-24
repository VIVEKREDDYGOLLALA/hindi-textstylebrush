import os
import cv2
import numpy as np
from PIL import Image
import torch
import re
from tqdm import tqdm
import argparse
from diffusers import StableDiffusionInpaintPipeline

# Conditionally import easyocr
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("EasyOCR not available. Installation: pip install easyocr")

# Conditionally import sklearn
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available. Installation: pip install scikit-learn")

def read_bbox_file(bbox_path):
    """
    Read the bounding box file with the specific format provided.
    
    Args:
        bbox_path: Path to the bounding box file
        
    Returns:
        List of bounding box data
    """
    bbox_data = []
    
    try:
        with open(bbox_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Handle the first line with dimensions
            if line.startswith('[') and line.endswith(']') and ',' in line and 'paragraph' not in line and 'figure' not in line:
                try:
                    # This is likely the dimensions line [width, height]
                    values = line.strip('[]').split(',')
                    width = int(values[0])
                    height = int(values[1])
                    bbox_data.append(['dimensions', [width, height]])
                    continue
                except:
                    pass
            
            # Special handling for the specific format provided
            # Extract type, coordinates, and ID using regex
            match = re.match(r'\[(.*?), \[(.*?)\], (.*?), (.*?)\]', line)
            if match:
                bbox_type = match.group(1)
                coords_str = match.group(2)
                id1 = match.group(3)
                id2 = match.group(4)
                
                # Parse coordinates
                coords = [float(x) for x in coords_str.split(',')]
                if len(coords) >= 4:
                    x, y, width, height = coords
                    bbox_data.append([bbox_type, [x, y, width, height], id1, id2])
    except Exception as e:
        print(f"Error reading file {bbox_path}: {e}")
        
    return bbox_data

def get_background_description(box_roi, mask):
    """
    Determine the dominant background colors for better prompting
    
    Args:
        box_roi: Region of interest from the image
        mask: Mask of text areas
        
    Returns:
        String description of the background
    """
    if not SKLEARN_AVAILABLE:
        return "seamless background"
        
    # Exclude masked areas when calculating dominant colors
    inverse_mask = cv2.bitwise_not(mask)
    background_pixels = cv2.bitwise_and(box_roi, box_roi, mask=inverse_mask)
    
    # Flatten and remove black (masked) pixels
    pixels = background_pixels.reshape(-1, 3)
    pixels = pixels[~np.all(pixels == 0, axis=1)]
    
    # If we have background pixels, determine if it's colorful or simple
    background_description = "simple neutral background"
    if len(pixels) > 0:
        # Calculate color variance
        color_std = np.std(pixels, axis=0).mean()
        if color_std > 30:
            background_description = "colorful textured background"
        else:
            # Get dominant color
            if len(pixels) > 1000:  # Subsample for speed
                pixels = pixels[np.random.choice(len(pixels), 1000, replace=False)]
            
            kmeans = KMeans(n_clusters=1, n_init=10)
            kmeans.fit(pixels)
            dominant_color = kmeans.cluster_centers_[0]
            
            # Determine general color name
            if np.all(dominant_color > 200):
                background_description = "white background"
            elif np.mean(dominant_color) < 50:
                background_description = "dark background"
            elif dominant_color[0] > 150 and dominant_color[1] < 100 and dominant_color[2] < 100:
                background_description = "red background"
            elif dominant_color[1] > 150 and dominant_color[0] < 100 and dominant_color[2] < 100:
                background_description = "green background"
            elif dominant_color[2] > 150 and dominant_color[0] < 100 and dominant_color[1] < 100:
                background_description = "blue background"
            elif dominant_color[2] > 100 and dominant_color[0] > 100:
                background_description = "purple background"
            elif dominant_color[0] > 100 and dominant_color[1] > 100:
                background_description = "yellow or orange background"
            else:
                background_description = "colored background"
                
    return background_description

def remove_text_from_image(image_path, bbox_data, output_path, use_ocr=True, use_ai=True):
    """
    Enhanced text removal using multiple detection methods and intelligent inpainting
    
    Args:
        image_path: Path to the input image
        bbox_data: List of bounding box data
        output_path: Path to save the processed image
        use_ocr: Whether to use OCR for text detection
        use_ai: Whether to use AI inpainting
        
    Returns:
        True if successful, False otherwise
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return False
        
    img_height, img_width = img.shape[:2]
    
    # Initialize EasyOCR reader if available and requested
    reader = None
    if use_ocr and EASYOCR_AVAILABLE:
        try:
            reader = easyocr.Reader(['en', 'bn', 'hi', 'ja', 'ch_tra', 'ar', 'fa', 'ur'])
            print("EasyOCR initialized with multiple languages")
        except Exception as e:
            print(f"Error initializing EasyOCR: {e}")
    
    # Initialize Stable Diffusion Inpainting model if requested
    pipe = None
    if use_ai:
        try:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = pipe.to(device)
            print(f"Stable Diffusion initialized on {device}")
        except Exception as e:
            print(f"Error initializing Stable Diffusion: {e}")
            use_ai = False
    
    # Process each box
    for item in bbox_data:
        if not isinstance(item, list) or len(item) < 2:
            continue
            
        label = item[0]
        
        # Skip dimensions entry and figure entries as they might contain main image content
        if label == "dimensions" or label == "figure":
            continue
            
        try:
            # Extract coordinates
            coords = item[1]
            if not isinstance(coords, list) or len(coords) < 4:
                continue
                
            x, y, w, h = [int(val) for val in coords]
            
            # Ensure coordinates are within bounds
            x = max(0, min(x, img_width-1))
            y = max(0, min(y, img_height-1))
            w = min(w, img_width-x)
            h = min(h, img_height-y)
            
            # Skip if box is too small
            if w < 10 or h < 10:
                continue
                
            # Get ID for logging, if available
            box_id = item[2] if len(item) > 2 else "unknown"
            
            print(f"Processing {label} box at [{x}, {y}, {w}, {h}], ID: {box_id}")
            
            # Extract region of interest
            box_roi = img[y:y+h, x:x+w].copy()
            
            # Create a combined mask using multiple methods
            # 1. Adaptive thresholding for text detection
            gray_roi = cv2.cvtColor(box_roi, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
            
            # 2. EasyOCR for text detection (if available)
            ocr_mask = np.zeros_like(gray_roi)
            if reader:
                try:
                    ocr_results = reader.readtext(box_roi)
                    for detection in ocr_results:
                        bbox = detection[0]
                        text_box = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(ocr_mask, [text_box], 255)
                except Exception as e:
                    print(f"OCR error for box {box_id}: {e}")
            
            # 3. MSER for connected component detection
            mser_mask = np.zeros_like(gray_roi)
            try:
                mser = cv2.MSER_create()
                regions, _ = mser.detectRegions(gray_roi)
                for region in regions:
                    hull = cv2.convexHull(region.reshape(-1, 1, 2))
                    cv2.fillPoly(mser_mask, [hull], 255)
            except Exception as e:
                print(f"MSER error for box {box_id}: {e}")
            
            # Combine all masks and clean up
            combined_mask = cv2.bitwise_or(binary, cv2.bitwise_or(ocr_mask, mser_mask))
            
            # Use color-based segmentation to refine mask for colored text
            hsv_roi = cv2.cvtColor(box_roi, cv2.COLOR_BGR2HSV)
            
            # Detect common text colors (black, white, strong colors)
            # Black text
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 50])
            black_mask = cv2.inRange(hsv_roi, lower_black, upper_black)
            
            # White text
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv_roi, lower_white, upper_white)
            
            # Combine color masks with the other detection methods
            color_mask = cv2.bitwise_or(black_mask, white_mask)
            
            # Create final mask
            kernel = np.ones((5, 5), np.uint8)
            combined_mask = cv2.bitwise_or(combined_mask, color_mask)
            final_mask = cv2.dilate(combined_mask, kernel, iterations=2)
            
            # If AI inpainting is available, use it
            if use_ai and pipe:
                try:
                    # Determine background description for better prompting
                    bg_desc = get_background_description(box_roi, final_mask)
                    
                    # Convert to PIL for inpainting
                    box_img_pil = Image.fromarray(cv2.cvtColor(box_roi, cv2.COLOR_BGR2RGB))
                    mask_pil = Image.fromarray(final_mask)
                    
                    # Inpaint with context-aware prompt
                    prompt = f"seamless {bg_desc} texture, no text, clean design"
                    
                    result = pipe(
                        prompt=prompt,
                        image=box_img_pil,
                        mask_image=mask_pil,
                        num_inference_steps=30,
                        guidance_scale=7.5,
                        negative_prompt="text, letters, writing, characters, watermark, signature"
                    ).images[0]
                    
                    # Convert back to OpenCV and place in original image
                    result_cv = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
                    img[y:y+h, x:x+w] = result_cv
                    
                except Exception as e:
                    print(f"AI inpainting error for box {box_id}: {e}")
                    # Fall back to OpenCV inpainting
                    result = cv2.inpaint(box_roi, final_mask, 10, cv2.INPAINT_NS)
                    img[y:y+h, x:x+w] = result
            else:
                # Use OpenCV inpainting as fallback
                result = cv2.inpaint(box_roi, final_mask, 10, cv2.INPAINT_NS)
                img[y:y+h, x:x+w] = result
                
        except Exception as e:
            print(f"Error processing box: {e}")
            continue
    
    # Save final result
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Successfully processed image. Result saved to {output_path}")
    return True

def process_bulk_images(images_dir, bboxes_dir, output_dir, use_ocr=True, use_ai=True):
    """
    Process multiple images using their corresponding bounding box files.
    
    Args:
        images_dir: Directory containing the images
        bboxes_dir: Directory containing bounding box files
        output_dir: Directory to save processed images
        use_ocr: Whether to use OCR for text detection
        use_ai: Whether to use AI inpainting
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    successful = 0
    failed = 0
    
    for image_file in tqdm(image_files, desc="Processing images"):
        # Construct paths
        image_path = os.path.join(images_dir, image_file)
        output_path = os.path.join(output_dir, image_file)
        
        # Try different possible extensions for bbox files
        base_name = os.path.splitext(image_file)[0]
        possible_extensions = ['.txt', '.json', '.bbox']
        
        bbox_path = None
        for ext in possible_extensions:
            potential_path = os.path.join(bboxes_dir, base_name + ext)
            if os.path.exists(potential_path):
                bbox_path = potential_path
                break
        
        # If no bbox file with matching name, try to use the first available
        if not bbox_path:
            bbox_files = [f for f in os.listdir(bboxes_dir) 
                         if f.lower().endswith(('.txt', '.json', '.bbox'))]
            if bbox_files:
                bbox_path = os.path.join(bboxes_dir, bbox_files[0])
                print(f"No matching bbox file found for {image_file}. Using {bbox_files[0]} instead.")
        
        # Process the image if bbox file found
        if bbox_path:
            # Read bounding boxes
            bbox_data = read_bbox_file(bbox_path)
            
            if bbox_data:
                success = remove_text_from_image(image_path, bbox_data, output_path, use_ocr, use_ai)
                if success:
                    successful += 1
                else:
                    failed += 1
            else:
                print(f"No valid bounding box data found in {bbox_path}")
                failed += 1
        else:
            print(f"Warning: No bounding box file found for {image_file}")
            failed += 1
    
    print(f"Processing complete. Successfully processed: {successful}, Failed: {failed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove text from images using bounding boxes with enhanced detection')
    parser.add_argument('--images', required=True, help='Directory containing images')
    parser.add_argument('--bboxes', required=True, help='Directory containing bounding box files')
    parser.add_argument('--output', required=True, help='Directory to save processed images')
    parser.add_argument('--no_ocr', action='store_true', help='Disable OCR-based text detection')
    parser.add_argument('--no_ai', action='store_true', help='Disable AI-based inpainting (use OpenCV instead)')
    
    args = parser.parse_args()
    
    # Process all images
    process_bulk_images(
        args.images, 
        args.bboxes, 
        args.output, 
        use_ocr=not args.no_ocr,
        use_ai=not args.no_ai
    )
