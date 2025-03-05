"""
Training script for the Hindi TextStyleBrush model.
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from config import Config
from data import get_dataloaders
from models import (
    StyleEncoder, ContentEncoder, StyleMappingNetwork,
    Generator, Discriminator, TypefaceClassifier,
    TextRecognizer
)
from loss import (
    PerceptualLoss, RecognitionLoss, ReconstructionLoss,
    CyclicReconstructionLoss, AdversarialLoss
)
from utils import save_images, plot_losses


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


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


def load_or_train_typeface_classifier(config, synth_loader):
    """
    Load a pre-trained typeface classifier or train a new one.
    
    Args:
        config: Configuration object
        synth_loader: DataLoader for synthetic font dataset
    
    Returns:
        Pre-trained TypefaceClassifier
    """
    classifier = TypefaceClassifier(num_classes=config.NUM_SYNTHETIC_FONTS).to(config.DEVICE)
    classifier_path = os.path.join(config.CHECKPOINT_DIR, 'typeface_classifier.pth')
    
    if os.path.exists(classifier_path):
        print(f"Loading pre-trained typeface classifier from {classifier_path}")
        classifier.load_state_dict(torch.load(classifier_path, map_location=config.DEVICE))
    else:
        print("Training typeface classifier...")
        # Set up optimizer
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Train the classifier
        classifier.train()
        num_epochs = 10
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch in tqdm(synth_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images = batch['image'].to(config.DEVICE)
                labels = batch['font_class'].to(config.DEVICE)
                
                optimizer.zero_grad()
                
                outputs = classifier(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(synth_loader):.4f}, Accuracy: {accuracy:.2f}%")
        
        # Save the trained classifier
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        torch.save(classifier.state_dict(), classifier_path)
        print(f"Saved typeface classifier to {classifier_path}")
    
    # Set classifier to evaluation mode
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False
    
    return classifier


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
        print("Initializing text recognizer with pre-trained weights...")
        # In a full implementation, you would load pre-trained weights for Hindi text recognition
        # Here we just initialize the model randomly
        pass
    
    # Set recognizer to evaluation mode
    recognizer.eval()
    for param in recognizer.parameters():
        param.requires_grad = False
    
    return recognizer


def save_checkpoint(models, optimizers, epoch, loss_dict, config, is_best=False):
    """
    Save model checkpoints.
    
    Args:
        models: Dictionary of models
        optimizers: Dictionary of optimizers
        epoch: Current epoch
        loss_dict: Dictionary of losses
        config: Configuration object
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'style_encoder': models['style_encoder'].state_dict(),
        'content_encoder': models['content_encoder'].state_dict(),
        'mapping_network': models['mapping_network'].state_dict(),
        'generator': models['generator'].state_dict(),
        'discriminator': models['discriminator'].state_dict(),
        'optimizer_G': optimizers['optimizer_G'].state_dict(),
        'optimizer_D': optimizers['optimizer_D'].state_dict(),
        'losses': loss_dict
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save latest checkpoint for resuming
    latest_path = os.path.join(config.CHECKPOINT_DIR, 'checkpoint_latest.pth')
    torch.save(checkpoint, latest_path)
    
    # Save best model if specified
    if is_best:
        best_path = os.path.join(config.CHECKPOINT_DIR, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")


def load_checkpoint(models, optimizers, config, checkpoint_path=None):
    """
    Load model checkpoint.
    
    Args:
        models: Dictionary of models
        optimizers: Dictionary of optimizers
        config: Configuration object
        checkpoint_path: Path to checkpoint file or None for latest
    
    Returns:
        start_epoch: Epoch to start from
        loss_dict: Dictionary of losses
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'checkpoint_latest.pth')
    
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found, starting from scratch")
        return 0, {}
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    
    # Load model states
    models['style_encoder'].load_state_dict(checkpoint['style_encoder'])
    models['content_encoder'].load_state_dict(checkpoint['content_encoder'])
    models['mapping_network'].load_state_dict(checkpoint['mapping_network'])
    models['generator'].load_state_dict(checkpoint['generator'])
    models['discriminator'].load_state_dict(checkpoint['discriminator'])
    
    # Load optimizer states
    optimizers['optimizer_G'].load_state_dict(checkpoint['optimizer_G'])
    optimizers['optimizer_D'].load_state_dict(checkpoint['optimizer_D'])
    
    return checkpoint['epoch'] + 1, checkpoint['losses']


def train_step(models, optimizers, loss_functions, data, config):
    """
    Execute one training step.
    
    Args:
        models: Dictionary of models
        optimizers: Dictionary of optimizers
        loss_functions: Dictionary of loss functions
        data: Data batch
        config: Configuration object
    
    Returns:
        gen_loss: Generator loss
        disc_loss: Discriminator loss
        losses: Dictionary of individual losses
    """
    # Extract models
    style_encoder = models['style_encoder']
    content_encoder = models['content_encoder']
    mapping_network = models['mapping_network']
    generator = models['generator']
    discriminator = models['discriminator']
    
    # Extract data
    localized_images = data['localized_image'].to(config.DEVICE)
    word_images = data['word_image'].to(config.DEVICE)
    content_images1 = data['content_image1'].to(config.DEVICE)
    content_images2 = data['content_image2'].to(config.DEVICE)
    text = data['text']  # Original text content
    content_text2 = data['content_text2']  # Second text content
    normalized_bbox = data['normalized_bbox'].to(config.DEVICE)
    
    # Extract loss functions
    perceptual_loss_fn = loss_functions['perceptual']
    recognition_loss_fn = loss_functions['recognition']
    reconstruction_loss_fn = loss_functions['reconstruction']
    cyclic_recon_loss_fn = loss_functions['cyclic']
    adversarial_loss_fn = loss_functions['adversarial']
    
    # ======== Train Discriminator ========
    optimizers['optimizer_D'].zero_grad()
    
    # Extract style features
    style_features = style_encoder(localized_images, normalized_bbox)
    
    # Extract content features
    content_features1 = content_encoder(content_images1)
    content_features2 = content_encoder(content_images2)
    
    # Generate style vectors for each layer via mapping network
    style_vectors = mapping_network(style_features)
    
    # Generate images and masks with original content
    gen_images1, gen_masks1 = generator(content_features1, style_vectors)
    
    # Generate images and masks with second content
    gen_images2, gen_masks2 = generator(content_features2, style_vectors)
    
    # Compute discriminator outputs
    real_outputs = discriminator(word_images)
    fake_outputs1 = discriminator(gen_images1.detach())
    
    # Compute discriminator loss
    disc_loss, disc_losses = adversarial_loss_fn.discriminator_loss(
        real_outputs=real_outputs,
        fake_outputs=fake_outputs1,
        real_images=word_images,
        apply_r1_reg=True
    )
    
    # Backpropagate discriminator loss
    disc_loss.backward()
    optimizers['optimizer_D'].step()
    
    # ======== Train Generator and encoders ========
    optimizers['optimizer_G'].zero_grad()
    
    # Compute discriminator outputs for generated images
    fake_outputs1 = discriminator(gen_images1)
    
    # Compute generator adversarial loss
    gen_adv_loss, gen_adv_losses = adversarial_loss_fn.generator_loss(
        fake_outputs=fake_outputs1,
        style_vectors=style_vectors,
        apply_pl_reg=True
    )
    
    # Compute perceptual loss
    percep_loss, percep_losses = perceptual_loss_fn(
        generated_images=gen_images1,
        target_images=word_images,
        lambda_per=config.LAMBDA_1,
        lambda_tex=config.LAMBDA_2,
        lambda_emb=config.LAMBDA_3
    )
    
    # Compute recognition loss (content preservation)
    recog_loss1 = recognition_loss_fn(gen_images1, text)
    recog_loss2 = recognition_loss_fn(gen_images2, content_text2)
    recog_loss = (recog_loss1 + recog_loss2) * config.LAMBDA_4
    
    # Compute reconstruction loss
    recon_loss, recon_losses = reconstruction_loss_fn(
        generated_images=gen_images1,
        target_images=word_images,
        masks=gen_masks1
    )
    recon_loss = recon_loss * config.LAMBDA_5
    
    # Cycle consistency: re-encode generated image and generate again
    style_features_cycle = style_encoder(gen_images1, normalized_bbox)
    style_vectors_cycle = mapping_network(style_features_cycle)
    gen_images_cycle, gen_masks_cycle = generator(content_features1, style_vectors_cycle)
    
    # Compute cyclic reconstruction loss
    cyclic_loss, cyclic_losses = cyclic_recon_loss_fn(
        original_images=gen_images1,
        cyclic_images=gen_images_cycle,
        masks=gen_masks1
    )
    cyclic_loss = cyclic_loss * config.LAMBDA_6
    
    # Combine all generator losses
    gen_loss = gen_adv_loss + percep_loss + recog_loss + recon_loss + cyclic_loss
    
    # Backpropagate generator loss
    gen_loss.backward()
    optimizers['optimizer_G'].step()
    
    # Collect all losses
    losses = {
        'generator': gen_loss.item(),
        'discriminator': disc_loss.item(),
        'adversarial': gen_adv_loss.item(),
        'perceptual': percep_loss.item(),
        'recognition': recog_loss.item(),
        'reconstruction': recon_loss.item(),
        'cyclic': cyclic_loss.item()
    }
    
    # Add detailed losses
    losses.update({f'gen_adv_{k}': v for k, v in gen_adv_losses.items()})
    losses.update({f'disc_{k}': v for k, v in disc_losses.items()})
    losses.update({f'percep_{k}': v for k, v in percep_losses.items()})
    losses.update({f'recon_{k}': v for k, v in recon_losses.items()})
    losses.update({f'cyclic_{k}': v for k, v in cyclic_losses.items()})
    
    return gen_loss.item(), disc_loss.item(), losses, {
        'gen_images1': gen_images1,
        'gen_masks1': gen_masks1,
        'gen_images2': gen_images2,
        'gen_masks2': gen_masks2
    }


def validate(models, loss_functions, val_loader, config, writer, epoch):
    """
    Validate the model.
    
    Args:
        models: Dictionary of models
        loss_functions: Dictionary of loss functions
        val_loader: Validation data loader
        config: Configuration object
        writer: TensorBoard writer
        epoch: Current epoch
    
    Returns:
        val_loss: Validation loss
    """
    print("\nValidating...")
    
    # Extract models
    style_encoder = models['style_encoder']
    content_encoder = models['content_encoder']
    mapping_network = models['mapping_network']
    generator = models['generator']
    discriminator = models['discriminator']
    
    # Set models to evaluation mode
    style_encoder.eval()
    content_encoder.eval()
    mapping_network.eval()
    generator.eval()
    discriminator.eval()
    
    val_losses = []
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader, desc="Validation")):
            # Extract data
            localized_images = data['localized_image'].to(config.DEVICE)
            word_images = data['word_image'].to(config.DEVICE)
            content_images1 = data['content_image1'].to(config.DEVICE)
            normalized_bbox = data['normalized_bbox'].to(config.DEVICE)
            
            # Extract style features
            style_features = style_encoder(localized_images, normalized_bbox)
            
            # Extract content features
            content_features1 = content_encoder(content_images1)
            
            # Generate style vectors for each layer
            style_vectors = mapping_network(style_features)
            
            # Generate images and masks
            gen_images1, gen_masks1 = generator(content_features1, style_vectors)
            
            # Compute reconstruction loss
            recon_loss, _ = loss_functions['reconstruction'](
                generated_images=gen_images1,
                target_images=word_images,
                masks=gen_masks1
            )
            
            val_losses.append(recon_loss.item())
            
            # Visualize generated images for the first few batches
            if i < 2:
                save_images(
                    images=gen_images1[:8],
                    masks=gen_masks1[:8],
                    original_images=word_images[:8],
                    target_images=content_images1[:8],
                    filename=os.path.join(config.OUTPUT_DIR, f'val_ep{epoch}_batch{i}.png')
                )
                
                # Log images to TensorBoard
                if writer is not None:
                    writer.add_images(f'Val/Original_Images', (word_images[:4] + 1) / 2, epoch)
                    writer.add_images(f'Val/Content_Images', (content_images1[:4] + 1) / 2, epoch)
                    writer.add_images(f'Val/Generated_Images', (gen_images1[:4] + 1) / 2, epoch)
                    writer.add_images(f'Val/Generated_Masks', gen_masks1[:4], epoch)
    
    # Set models back to training mode
    style_encoder.train()
    content_encoder.train()
    mapping_network.train()
    generator.train()
    discriminator.train()
    
    val_loss = sum(val_losses) / len(val_losses)
    print(f"Validation Loss: {val_loss:.4f}")
    
    # Log validation loss to TensorBoard
    if writer is not None:
        writer.add_scalar('Loss/Validation', val_loss, epoch)
    
    return val_loss


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Hindi TextStyleBrush model")
    parser.add_argument('--config', type=str, default='config.py', help='Path to config file')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Set random seed
    set_seed(config.SEED)
    
    # Create output directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=config.LOG_DIR)
    
    # Create character mapping
    char_to_idx, idx_to_char = create_char_map()
    
    # Get data loaders
    train_loader, val_loader, synth_loader = get_dataloaders(config)
    
    # Initialize or load pre-trained models
    typeface_classifier = load_or_train_typeface_classifier(config, synth_loader)
    text_recognizer = load_or_init_recognizer(config, char_to_idx)
    
    # Initialize models
    style_encoder = StyleEncoder(style_dim=config.STYLE_DIM).to(config.DEVICE)
    content_encoder = ContentEncoder(content_dim=config.CONTENT_DIM).to(config.DEVICE)
    mapping_network = StyleMappingNetwork(style_dim=config.STYLE_DIM).to(config.DEVICE)
    generator = Generator(content_dim=config.CONTENT_DIM, style_dim=config.STYLE_DIM).to(config.DEVICE)
    discriminator = Discriminator().to(config.DEVICE)
    
    # Initialize optimizers
    optimizer_G = optim.Adam(
        list(style_encoder.parameters()) + 
        list(content_encoder.parameters()) + 
        list(mapping_network.parameters()) + 
        list(generator.parameters()),
        lr=config.LEARNING_RATE,
        betas=config.BETAS
    )
    
    optimizer_D = optim.Adam(
        discriminator.parameters(),
        lr=config.LEARNING_RATE,
        betas=config.BETAS
    )
    
    # Create model and optimizer dictionaries
    models = {
        'style_encoder': style_encoder,
        'content_encoder': content_encoder,
        'mapping_network': mapping_network,
        'generator': generator,
        'discriminator': discriminator,
        'typeface_classifier': typeface_classifier,
        'text_recognizer': text_recognizer
    }
    
    optimizers = {
        'optimizer_G': optimizer_G,
        'optimizer_D': optimizer_D
    }
    
    # Initialize loss functions
    perceptual_loss = PerceptualLoss(typeface_classifier).to(config.DEVICE)
    recognition_loss = RecognitionLoss(text_recognizer, char_to_idx).to(config.DEVICE)
    reconstruction_loss = ReconstructionLoss().to(config.DEVICE)
    cyclic_recon_loss = CyclicReconstructionLoss().to(config.DEVICE)
    adversarial_loss = AdversarialLoss().to(config.DEVICE)
    
    loss_functions = {
        'perceptual': perceptual_loss,
        'recognition': recognition_loss,
        'reconstruction': reconstruction_loss,
        'cyclic': cyclic_recon_loss,
        'adversarial': adversarial_loss
    }
    
    # Resume training if specified
    start_epoch = 0
    best_val_loss = float('inf')
    loss_history = {
        'generator': [],
        'discriminator': [],
        'validation': []
    }
    
    if args.resume or args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        start_epoch, loss_dict = load_checkpoint(models, optimizers, config, checkpoint_path)
        
        if 'validation' in loss_dict:
            loss_history = loss_dict
            best_val_loss = min(loss_dict['validation']) if loss_dict['validation'] else float('inf')
    
    # Training loop
    num_epochs = config.NUM_EPOCHS
    
    print(f"Starting training from epoch {start_epoch} for {num_epochs} epochs")
    
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        
        # Training
        style_encoder.train()
        content_encoder.train()
        mapping_network.train()
        generator.train()
        discriminator.train()
        
        epoch_gen_losses = []
        epoch_disc_losses = []
        all_losses = {}
        
        for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Training step
            gen_loss, disc_loss, losses, outputs = train_step(
                models, optimizers, loss_functions, data, config
            )
            
            # Collect losses
            epoch_gen_losses.append(gen_loss)
            epoch_disc_losses.append(disc_loss)
            
            # Accumulate detailed losses
            for k, v in losses.items():
                if k not in all_losses:
                    all_losses[k] = []
                all_losses[k].append(v)
            
            # Log to TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            for k, v in losses.items():
                writer.add_scalar(f'Train/{k}', v, global_step)
            
            # Visualize generated images periodically
            if batch_idx % 100 == 0:
                save_images(
                    images=outputs['gen_images1'][:8],
                    masks=outputs['gen_masks1'][:8],
                    original_images=data['word_image'][:8].to(config.DEVICE),
                    target_images=data['content_image1'][:8].to(config.DEVICE),
                    filename=os.path.join(config.OUTPUT_DIR, f'train_ep{epoch}_batch{batch_idx}.png')
                )
        
        # Calculate average losses for the epoch
        avg_gen_loss = sum(epoch_gen_losses) / len(epoch_gen_losses)
        avg_disc_loss = sum(epoch_disc_losses) / len(epoch_disc_losses)
        avg_losses = {k: sum(v) / len(v) for k, v in all_losses.items()}
        
        # Log epoch losses to TensorBoard
        writer.add_scalar('Loss/Generator', avg_gen_loss, epoch)
        writer.add_scalar('Loss/Discriminator', avg_disc_loss, epoch)
        
        # Update loss history
        loss_history['generator'].append(avg_gen_loss)
        loss_history['discriminator'].append(avg_disc_loss)
        
        # Validation
        val_loss = validate(models, loss_functions, val_loader, config, writer, epoch)
        loss_history['validation'].append(val_loss)
        
        # Save plots of losses
        plot_losses(loss_history, os.path.join(config.OUTPUT_DIR, 'losses.png'))
        
        # Check if this is the best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        # Save checkpoint
        if (epoch + 1) % config.SAVE_INTERVAL == 0 or is_best:
            save_checkpoint(models, optimizers, epoch, loss_history, config, is_best)
        
        # Print epoch summary
        time_elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} completed in {time_elapsed:.2f}s")
        print(f"Generator Loss: {avg_gen_loss:.4f}, Discriminator Loss: {avg_disc_loss:.4f}, Validation Loss: {val_loss:.4f}")
    
    # Save final model
    save_checkpoint(models, optimizers, num_epochs-1, loss_history, config)
    
    # Close TensorBoard writer
    writer.close()
    
    print("Training complete!")


if __name__ == "__main__":
    main()# Save the trained classifier
        # os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        # torch.save(classifier.state_dict(), classifier_path)
        # print(f"Saved typeface classifier to {classifier_path}")
    
    # Set classifier to evaluation mode
    # classifier.eval()
    # for param in classifier.parameters():
    #     param.requires_gra=False