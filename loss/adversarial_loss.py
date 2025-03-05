"""
Adversarial losses for the Hindi TextStyleBrush model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NonSaturatingGANLoss(nn.Module):
    """
    Non-saturating GAN loss for generator training.
    Helps to avoid vanishing gradients.
    """
    def __init__(self):
        super(NonSaturatingGANLoss, self).__init__()
    
    def forward(self, discriminator_output, is_real=True):
        """
        Compute non-saturating GAN loss.
        
        Args:
            discriminator_output: Output of the discriminator [B, 1, H, W]
            is_real: Whether the target is real (True) or fake (False)
        
        Returns:
            loss: GAN loss
        """
        if is_real:
            # For real samples: minimize -log(D(x))
            loss = F.softplus(-discriminator_output).mean()
        else:
            # For fake samples: minimize -log(1-D(G(z))) -> maximize log(D(G(z)))
            loss = F.softplus(discriminator_output).mean()
        
        return loss


class R1Regularization(nn.Module):
    """
    R1 gradient penalty for discriminator regularization.
    Penalizes the gradient norm of the discriminator with respect to real samples.
    """
    def __init__(self, gamma=10.0):
        super(R1Regularization, self).__init__()
        self.gamma = gamma
    
    def forward(self, real_images, discriminator_real_outputs):
        """
        Compute R1 gradient penalty.
        
        Args:
            real_images: Real images [B, 3, H, W]
            discriminator_real_outputs: Discriminator outputs for real images [B, 1, H', W']
        
        Returns:
            penalty: R1 gradient penalty
        """
        # Create graph for gradients
        grad_real = torch.autograd.grad(
            outputs=discriminator_real_outputs.sum(),
            inputs=real_images,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute gradient penalty
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
        
        # Apply gamma scaling
        penalty = self.gamma * 0.5 * grad_penalty
        
        return penalty


class PathLengthRegularization(nn.Module):
    """
    Path length regularization for generator training.
    Encourages smooth mapping from latent space to image space.
    """
    def __init__(self, beta=0.99, decay=0.01):
        super(PathLengthRegularization, self).__init__()
        self.beta = beta
        self.decay = decay
        self.register_buffer('path_length_mean', torch.tensor(0.0))
    
    def forward(self, generated_images, style_vectors):
        """
        Compute path length regularization.
        
        Args:
            generated_images: Generated images [B, 3, H, W]
            style_vectors: Style vectors that produced the images (list of tensors)
        
        Returns:
            penalty: Path length regularization penalty
        """
        # Compute noise vector for jittering
        batch_size = generated_images.shape[0]
        noise = torch.randn_like(generated_images) / np.sqrt(generated_images.shape[2] * generated_images.shape[3])
        
        # Calculate gradients
        # Combine style vectors for simplicity
        combined_style = torch.cat([s.view(batch_size, -1) for s in style_vectors], dim=1)
        
        # Calculate gradients with respect to combined style
        gradients = torch.autograd.grad(
            outputs=[(generated_images * noise).sum()],
            inputs=[combined_style],
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute path length penalty
        path_lengths = torch.sqrt(gradients.pow(2).sum(1) + 1e-8)
        
        # Update path length mean (running average)
        path_length_mean = self.path_length_mean.lerp(path_lengths.mean(), self.decay)
        self.path_length_mean.copy_(path_length_mean.detach())
        
        # Calculate penalty
        penalty = (path_lengths - path_length_mean).pow(2).mean()
        
        return penalty


class AdversarialLoss(nn.Module):
    """
    Combined adversarial losses for generator and discriminator.
    Includes non-saturating GAN loss, R1 regularization, and path length regularization.
    """
    def __init__(self, gamma_r1=10.0, pl_weight=2.0):
        super(AdversarialLoss, self).__init__()
        self.gan_loss = NonSaturatingGANLoss()
        self.r1_reg = R1Regularization(gamma=gamma_r1)
        self.pl_reg = PathLengthRegularization()
        self.pl_weight = pl_weight
    
    def generator_loss(self, fake_outputs, style_vectors=None, apply_pl_reg=False):
        """
        Compute generator adversarial loss.
        
        Args:
            fake_outputs: Discriminator outputs for fake images [B, 1, H', W']
            style_vectors: Style vectors (for path length regularization)
            apply_pl_reg: Whether to apply path length regularization
        
        Returns:
            total_loss: Total generator loss
            losses: Dictionary of individual losses
        """
        # Basic GAN loss
        gen_loss = self.gan_loss(fake_outputs, is_real=False)
        
        losses = {'gen_loss': gen_loss.item()}
        total_loss = gen_loss
        
        # Path length regularization (if enabled and style vectors provided)
        if apply_pl_reg and style_vectors is not None:
            pl_penalty = self.pl_reg(fake_outputs, style_vectors)
            losses['pl_penalty'] = pl_penalty.item()
            total_loss = total_loss + self.pl_weight * pl_penalty
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses
    
    def discriminator_loss(self, real_outputs, fake_outputs, real_images, apply_r1_reg=True):
        """
        Compute discriminator adversarial loss.
        
        Args:
            real_outputs: Discriminator outputs for real images [B, 1, H', W']
            fake_outputs: Discriminator outputs for fake images [B, 1, H', W']
            real_images: Real images (for R1 regularization) [B, 3, H, W]
            apply_r1_reg: Whether to apply R1 regularization
        
        Returns:
            total_loss: Total discriminator loss
            losses: Dictionary of individual losses
        """
        # Real and fake GAN losses
        real_loss = self.gan_loss(real_outputs, is_real=True)
        fake_loss = self.gan_loss(fake_outputs, is_real=False)
        
        # Basic discriminator loss
        disc_loss = real_loss + fake_loss
        
        losses = {
            'disc_real_loss': real_loss.item(),
            'disc_fake_loss': fake_loss.item(),
            'disc_loss': disc_loss.item()
        }
        
        total_loss = disc_loss
        
        # R1 regularization (if enabled)
        if apply_r1_reg:
            r1_penalty = self.r1_reg(real_images, real_outputs)
            losses['r1_penalty'] = r1_penalty.item()
            total_loss = total_loss + r1_penalty
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses