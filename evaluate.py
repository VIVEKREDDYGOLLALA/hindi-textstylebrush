"""
Evaluation script for the Hindi TextStyleBrush model.
"""

import os
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from config import Config
from data import get_dataloaders
from models import StyleEncoder, ContentEncoder, StyleMappingNetwork, Generator, TextRecognizer
from utils import save_images, mse, psnr, ssim, text_recognition_accuracy, character_error_rate, word_error_rate


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


def create_char_map():
    """Create character to index mapping for Hindi characters."""
    char_to_idx = {}
    idx_to_char = {}
    
    # Add characters from config
    for i, char in enumerate(Config.ALL_CHARS):
        char_to_idx[char] = i
        idx_to_char[i] = char
    
    # Add special tokens
    char_to_idx['<pad>'] = len(char_to_idx)
    idx_to_char[len(char_to_idx) - 1] = '<pad>'
    
    char_to_idx['<eos>'] = len(char_to_idx)
    idx_to_char[len(char_to_idx) - 1] = '<eos>'
    
    char_to_idx['<unk>'] = len(char_to_idx)
    idx_to_char[len(char_to_idx) - 1] = '<unk>'
    
    return char_to_idx, idx_to_char


def load_or_init_recognizer(config, char_to_idx):
    """
    Load a pre-trained text recognizer or initialize a new one.
    
    Args:
        config: Configuration object
        char_to_idx: Character to index mapping
    
    Returns:
        Pre-trained TextRecognizer
    """
    num_classes = len(char_to_idx)
    recognizer = TextRecognizer(num_classes=num_classes).to(config.DEVICE)
    recognizer_path = os.path.join(config.CHECKPOINT_DIR, 'text_recognizer.pth')
    
    if os.path.exists(recognizer_path):
        print(f"Loading pre-trained text recognizer from {recognizer_path}")
        recognizer.load_state_dict(torch.load(recognizer_path, map_location=config.DEVICE))
    else:
        print("WARNING: No pre-trained text recognizer found. Using random initialization.")
    
    # Set recognizer to evaluation mode
    recognizer.eval()
    for param in recognizer.parameters():
        param.requires_grad = False
    
    return recognizer


def decode_predictions(predictions, idx_to_char):
    """
    Decode model predictions to text strings.
    
    Args:
        predictions: Model predictions tensor [B, T, C]
        idx_to_char: Index to character mapping
    
    Returns:
        List of predicted text strings
    """
    # Get the most likely character at each position
    _, predicted_indices = predictions.max(2)
    
    # Convert to CPU and numpy
    predicted_indices = predicted_indices.cpu().numpy()
    
    # Decode the predictions
    texts = []
    for indices in predicted_indices:
        # Find EOS token if present
        eos_pos = np.where(indices == idx_to_char['<eos>'])[0]
        if len(eos_pos) > 0:
            indices = indices[:eos_pos[0]]
        
        # Convert indices to characters and join
        text = ''.join([idx_to_char.get(idx, '') for idx in indices if idx not in 
                        [idx_to_char.get('<pad>', -1), idx_to_char.get('<eos>', -1)]])
        texts.append(text)
    
    return texts


def evaluate(models, dataloader, recognizer, char_to_idx, idx_to_char, config, output_dir):
    """
    Evaluate the model on the given dataset.
    
    Args:
        models: Dict of models
        dataloader: DataLoader for evaluation
        recognizer: Text recognizer model
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        config: Configuration object
        output_dir: Directory to save outputs
    
    Returns:
        Dict of evaluation metrics
    """
    style_encoder = models['style_encoder']
    content_encoder = models['content_encoder']
    mapping_network = models['mapping_network']
    generator = models['generator']
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Metrics
    metrics = {
        'mse': [],
        'psnr': [],
        'ssim': [],
        'accuracy': [],
        'cer': [],
        'wer': []
    }
    
    # Keep track of all predictions
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Extract data
            localized_images = data['localized_image'].to(config.DEVICE)
            word_images = data['word_image'].to(config.DEVICE)
            content_images1 = data['content_image1'].to(config.DEVICE)
            content_images2 = data['content_image2'].to(config.DEVICE)
            text = data['text']  # Original text content
            content_text2 = data['content_text2']  # Second text content
            normalized_bbox = data['normalized_bbox'].to(config.DEVICE)
            
            # Extract style features
            style_features = style_encoder(localized_images, normalized_bbox)
            
            # Extract content features
            content_features1 = content_encoder(content_images1)
            content_features2 = content_encoder(content_images2)
            
            # Generate style vectors for each layer
            style_vectors = mapping_network(style_features)
            
            # Generate images and masks with original content
            gen_images1, gen_masks1 = generator(content_features1, style_vectors)
            
            # Generate images and masks with second content
            gen_images2, gen_masks2 = generator(content_features2, style_vectors)
            
            # Save a few examples
            if batch_idx < 5:
                save_images(
                    images=gen_images1[:8],
                    masks=gen_masks1[:8],
                    original_images=word_images[:8],
                    target_images=content_images1[:8],
                    filename=os.path.join(output_dir, f'eval_batch{batch_idx}_content1.png')
                )
                
                save_images(
                    images=gen_images2[:8],
                    masks=gen_masks2[:8],
                    original_images=word_images[:8],
                    target_images=content_images2[:8],
                    filename=os.path.join(output_dir, f'eval_batch{batch_idx}_content2.png')
                )
            
            # Calculate pixel-level metrics for content 1
            metrics['mse'].append(mse(gen_images1, word_images, gen_masks1))
            metrics['psnr'].append(psnr(gen_images1, word_images, gen_masks1))
            metrics['ssim'].append(ssim(gen_images1, word_images, gen_masks1))
            
            # Recognize text in generated images
            recognized_gen1 = recognizer(gen_images1)
            recognized_gen2 = recognizer(gen_images2)
            
            # Decode predictions
            pred_texts1 = decode_predictions(recognized_gen1, idx_to_char)
            pred_texts2 = decode_predictions(recognized_gen2, idx_to_char)
            
            # Calculate text recognition metrics
            metrics['accuracy'].append(text_recognition_accuracy(pred_texts1, text))
            metrics['cer'].append(character_error_rate(pred_texts1, text))
            metrics['wer'].append(word_error_rate(pred_texts1, text))
            
            # Add predictions to lists
            all_predictions.extend(pred_texts1 + pred_texts2)
            all_targets.extend(text + content_text2)
    
    # Calculate average metrics
    avg_metrics = {k: sum(v) / len(v) if len(v) > 0 else 0 for k, v in metrics.items()}
    
    # Overall text recognition accuracy
    overall_accuracy = text_recognition_accuracy(all_predictions, all_targets)
    overall_cer = character_error_rate(all_predictions, all_targets)
    overall_wer = word_error_rate(all_predictions, all_targets)
    
    avg_metrics['overall_accuracy'] = overall_accuracy
    avg_metrics['overall_cer'] = overall_cer
    avg_metrics['overall_wer'] = overall_wer
    
    # Save results to CSV
    results_df = pd.DataFrame([avg_metrics])
    results_df.to_csv(os.path.join(output_dir, 'evaluation_metrics.csv'), index=False)
    
    # Print results
    print("\nEvaluation Results:")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")
    
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hindi TextStyleBrush model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory (overrides config)')
    parser.add_argument('--output_dir', type=str, default='eval_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Override config with command line arguments
    if args.data_dir:
        config.DATA_DIR = args.data_dir
    
    config.BATCH_SIZE = args.batch_size
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create character mappings
    char_to_idx, idx_to_char = create_char_map()
    
    # Load data loaders
    _, val_loader, _ = get_dataloaders(config)
    
    # Load models
    models = load_models(config, args.checkpoint)
    
    # Load text recognizer
    recognizer = load_or_init_recognizer(config, char_to_idx)
    
    # Evaluate models
    metrics = evaluate(models, val_loader, recognizer, char_to_idx, idx_to_char, config, args.output_dir)
    
    # Save detailed metrics
    with open(os.path.join(args.output_dir, 'evaluation_summary.txt'), 'w') as f:
        f.write("Hindi TextStyleBrush Evaluation Results\n")
        f.write(f"Checkpoint: {args.checkpoint}\n\n")
        
        f.write("Metrics:\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    
    print(f"Evaluation completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()