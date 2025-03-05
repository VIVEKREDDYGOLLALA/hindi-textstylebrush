"""
Inference script for the Hindi TextStyleBrush model.
"""

import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from torchvision import transforms

from config import Config
from models import StyleEncoder, ContentEncoder, StyleMappingNetwork, Generator
from utils import tensor_to_image


def load_models(config, checkpoint_path):
    """
    Load trained models from checkpoint.
    
    Args:
        config: Configuration object
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Dict of models
    """
    # Initialize models
    style_encoder = StyleEncoder(style_dim=config.STYLE_DIM).to(config.DEVICE)
    content_encoder = ContentEncoder(content_dim=config.CONTENT_DIM).to(config.DEVICE)
    mapping_network = StyleMappingNetwork(style_dim=config.STYLE_DIM).to(config.DEVICE)
    generator = Generator(content_dim=config.CONTENT_DIM, style_dim=config.STYLE_DIM).to(config.DEVICE)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    
    # Load model states
    style_encoder.load_state_dict(checkpoint['style_encoder'])
    content_encoder.load_state_dict(checkpoint['content_encoder'])
    mapping_network.load_state_dict(checkpoint['mapping_network'])
    generator.load_state_dict(checkpoint['generator'])
    
    # Set models to evaluation mode
    style_encoder.eval()
    content_encoder.eval()
    mapping_network.eval()
    generator.eval()
    
    return {
        'style_encoder': style_encoder,
        'content_encoder': content_encoder,
        'mapping_network': mapping_network,
        'generator': generator
    }


def load_image(image_path, transform=None):
    """
    Load an image from path and apply optional transforms.
    
    Args:
        image_path: Path to the image
        transform: Optional transforms to apply
    
    Returns:
        Transformed image tensor
    """
    image = Image.open(image_path).convert('RGB')
    
    if transform:
        image = transform(image)
    
    return image


def render_content_image(text, font_path=None, image_size=(64, 256)):
    """
    Render the content text using a standard Hindi font.
    
    Args:
        text: Text content string
        font_path: Path to Hindi font file (optional)
        image_size: Size of the output image (height, width)
    
    Returns:
        Rendered content image as tensor
    """
    # Create blank image
    img = Image.new('L', (image_size[1], image_size[0]), color=255)
    draw = ImageDraw.Draw(img)
    
    # Load font
    if font_path and os.path.exists(font_path):
        try:
            font = ImageFont.truetype(font_path, 32)
        except Exception as e:
            print(f"Error loading font: {e}")
            font = ImageFont.load_default()
    else:
        # Try to find a default Hindi font
        try:
            # Look for common Hindi fonts on the system
            common_hindi_fonts = [
                "Nirmala UI", "Mangal", "Lohit Devanagari",
                "Noto Sans Devanagari", "Gargi", "Chandas"
            ]
            
            font = None
            for font_name in common_hindi_fonts:
                try:
                    font = ImageFont.truetype(font_name, 32)
                    break
                except:
                    continue
            
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
    
    # Calculate text position to center it
    try:
        text_width, text_height = draw.textsize(text, font=font)
    except:
        # Fallback for newer Pillow versions
        text_width, text_height = font.getsize(text)
    
    position = ((image_size[1] - text_width) // 2, (image_size[0] - text_height) // 2)
    
    # Draw text
    draw.text(position, text, font=font, fill=0)
    
    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    return transform(img).unsqueeze(0)  # Add batch dimension


def process_style_image(image_path, bbox=None, transform=None):
    """
    Process style image and extract the text region.
    
    Args:
        image_path: Path to style image
        bbox: Text bounding box [x1, y1, x2, y2] or None for the whole image
        transform: Optional transforms to apply
    
    Returns:
        style_image: Processed style image tensor
        normalized_bbox: Normalized bounding box coordinates
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    height, width = image.shape[:2]
    
    # If no bounding box provided, use the whole image
    if bbox is None:
        # Add a small margin
        margin = min(height, width) // 10
        bbox = [margin, margin, width - margin, height - margin]
    
    # Extract text region with context
    x1, y1, x2, y2 = bbox
    
    # Add context around the bbox (20% padding)
    w, h = x2 - x1, y2 - y1
    context_x1 = max(0, x1 - int(w * 0.2))
    context_y1 = max(0, y1 - int(h * 0.2))
    context_x2 = min(width, x2 + int(w * 0.2))
    context_y2 = min(height, y2 + int(h * 0.2))
    
    # Extract region with context
    region = image[context_y1:context_y2, context_x1:context_x2]
    
    # Convert to PIL Image
    region_pil = Image.fromarray(region)
    
    # Apply transform if provided
    if transform:
        region_tensor = transform(region_pil).unsqueeze(0)  # Add batch dimension
    else:
        transform = transforms.Compose([
            transforms.Resize((Config.STYLE_IMAGE_SIZE, Config.STYLE_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        region_tensor = transform(region_pil).unsqueeze(0)
    
    # Calculate normalized bbox coordinates relative to the region with context
    norm_x1 = (x1 - context_x1) / (context_x2 - context_x1)
    norm_y1 = (y1 - context_y1) / (context_y2 - context_y1)
    norm_x2 = (x2 - context_x1) / (context_x2 - context_x1)
    norm_y2 = (y2 - context_y1) / (context_y2 - context_y1)
    
    normalized_bbox = torch.tensor([[norm_x1, norm_y1, norm_x2, norm_y2]], dtype=torch.float32)
    
    return region_tensor, normalized_bbox


def transfer_text_style(models, style_image, normalized_bbox, content_image, config):
    """
    Transfer text style from style image to content image.
    
    Args:
        models: Dict of models
        style_image: Style image tensor [1, 3, H, W]
        normalized_bbox: Normalized bounding box coordinates [1, 4]
        content_image: Content image tensor [1, 1, H, W]
        config: Configuration object
    
    Returns:
        generated_image: Generated image tensor [1, 3, H, W]
        mask: Generated mask tensor [1, 1, H, W]
    """
    style_encoder = models['style_encoder']
    content_encoder = models['content_encoder']
    mapping_network = models['mapping_network']
    generator = models['generator']
    
    # Move inputs to device
    style_image = style_image.to(config.DEVICE)
    normalized_bbox = normalized_bbox.to(config.DEVICE)
    content_image = content_image.to(config.DEVICE)
    
    with torch.no_grad():
        # Extract style features
        style_features = style_encoder(style_image, normalized_bbox)
        
        # Extract content features
        content_features = content_encoder(content_image)
        
        # Generate style vectors for each layer
        style_vectors = mapping_network(style_features)
        
        # Generate image and mask
        generated_image, mask = generator(content_features, style_vectors)
    
    return generated_image, mask


def blend_into_original(original_image_path, generated_image, mask, bbox, output_path):
    """
    Blend the generated text image back into the original image.
    
    Args:
        original_image_path: Path to original image
        generated_image: Generated image tensor [1, 3, H, W]
        mask: Generated mask tensor [1, 1, H, W]
        bbox: Target bounding box [x1, y1, x2, y2]
        output_path: Path to save the blended image
    """
    # Load original image
    original = cv2.imread(original_image_path)
    if original is None:
        raise ValueError(f"Could not load image from {original_image_path}")
    
    # Convert generated image and mask to numpy
    gen_image = tensor_to_image(generated_image[0])
    gen_mask = tensor_to_image(mask[0])
    
    # Ensure the generated image is in BGR format for OpenCV
    if gen_image.ndim == 3 and gen_image.shape[2] == 3:
        gen_image = cv2.cvtColor(gen_image, cv2.COLOR_RGB2BGR)
    
    # Resize generated image to fit the target bbox
    x1, y1, x2, y2 = bbox
    target_height, target_width = y2 - y1, x2 - x1
    
    resized_image = cv2.resize(gen_image, (target_width, target_height))
    resized_mask = cv2.resize(gen_mask, (target_width, target_height))
    
    # If mask is grayscale, ensure it has 3 channels
    if resized_mask.ndim == 2:
        resized_mask = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)
    
    # Normalize mask to [0, 1]
    resized_mask = resized_mask / 255.0
    
    # Create ROI in the original image
    roi = original[y1:y2, x1:x2]
    
    # Blend images using the mask
    blended = (resized_image * resized_mask + roi * (1 - resized_mask)).astype(np.uint8)
    
    # Replace the ROI in the original image
    result = original.copy()
    result[y1:y2, x1:x2] = blended
    
    # Save the result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result)
    
    print(f"Saved blended image to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Hindi TextStyleBrush Inference")
    parser.add_argument('--style_image', type=str, required=True, help='Path to style image')
    parser.add_argument('--bbox', type=int, nargs=4, help='Bounding box [x1, y1, x2, y2]')
    parser.add_argument('--content', type=str, required=True, help='Hindi text content to generate')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='output.png', help='Output image path')
    parser.add_argument('--font', type=str, help='Path to Hindi font for content rendering')
    parser.add_argument('--blend', action='store_true', help='Blend generated text into original image')
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    # Load models
    models = load_models(config, args.checkpoint)
    
    # Process style image
    style_image, normalized_bbox = process_style_image(args.style_image, args.bbox)
    
    # Render content image
    content_image = render_content_image(args.content, args.font)
    
    # Transfer text style
    generated_image, mask = transfer_text_style(models, style_image, normalized_bbox, content_image, config)
    
    # Save the generated image
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensor to image and save
    gen_image = tensor_to_image(generated_image[0])
    gen_mask = tensor_to_image(mask[0])
    
    # Save generated image and mask
    base_name, ext = os.path.splitext(args.output)
    cv2.imwrite(f"{base_name}_gen{ext}", cv2.cvtColor(gen_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{base_name}_mask{ext}", gen_mask)
    
    print(f"Saved generated image to {base_name}_gen{ext}")
    print(f"Saved mask to {base_name}_mask{ext}")
    
    # Blend into original image if requested
    if args.blend and args.bbox:
        blend_into_original(args.style_image, generated_image, mask, args.bbox, args.output)


if __name__ == "__main__":
    main()