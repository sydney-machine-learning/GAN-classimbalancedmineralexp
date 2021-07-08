import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class train_discriminator():
  def __init__(self, real_data, SMOTE_data, opt_d, generator, discriminator, device):
    self.real_data = real_data
    self.SMOTE_data = SMOTE_data
    self.opt_d = opt_d
    self.discriminator = discriminator
    self.generator = generator
    self.device = device

  def __call__(self):
    self.opt_d.zero_grad()

    # Pass real data through discriminator
    real_preds = self.discriminator(self.real_data)
    real_targets = torch.ones_like(real_preds, device=self.device)
    #real_loss = F.binary_cross_entropy(real_preds, real_targets)
    criterion = nn.BCEWithLogitsLoss()
    real_loss = criterion(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()
    
    # Generate fake data
    #latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_data = self.generator(self.SMOTE_data)
    #fake = gen(X_oversampled.float().to(device))

    # Pass fake data through discriminator    
    fake_preds = self.discriminator(fake_data)
    fake_targets = torch.zeros_like(fake_preds, device=self.device)
    #fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)

    fake_loss = criterion(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = (real_loss + fake_loss)/2
    loss.backward()
    self.opt_d.step()
    return loss.item(), real_score, fake_score


class train_generator():
    def __init__(self, SMOTE_data, opt_g, generator, discriminator, device):
        self.SMOTE_data = SMOTE_data
        self.opt_g = opt_g
        self.generator = generator
        self.discriminator = discriminator
        self.device = device

    def __call__(self):
        # Clear generator gradients
        self.opt_g.zero_grad()
        
        # Generate fake images
        fake_data = self.generator(self.SMOTE_data)
        
        # Try to fool the discriminator
        preds = self.discriminator(fake_data)
        targets = torch.ones_like(preds, device=self.device)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(preds, targets)
        
        # Update generator weights
        loss.backward()
        self.opt_g.step()
        
        return loss.item()