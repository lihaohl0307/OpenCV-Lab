# FGSM
import cv2
import numpy as np
import torchvision

# Load original image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
# Create noise
noise = np.random.normal(0, 25, image.shape).astype('uint8')
# Add noise to create adversarial image
adversarial_image = cv2.addWeighted(image, 1.0, noise, 0.1, 0)
cv2.imshow('Adversarial Image', adversarial_image)
cv2.waitKey(0)


# Diffusion Models

# Load and preprocess the image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (128, 128)) / 255.0 # Normalize to range [0, 1]
# Add Gaussian noise in steps
for i in range(5): # 5 noise addition steps
    noise = np.random.normal(0, 0.1 * (i + 1), image.shape) #Increase noise level
    noisy_image = np.clip(image + noise, 0, 1) # Ensure valuesare within [0, 1]
    cv2.imshow(f"Step {i+1}", (noisy_image *255).astype('uint8')) # Show noisy image

cv2.waitKey(0) # Wait for user input to close the windows

# Denoising using GaussianBlur
denoised_image = cv2.GaussianBlur(noisy_image, (5, 5), 0)
cv2.imshow("Denoised Image", (denoised_image *
255).astype('uint8'))
cv2.waitKey(0)



# Generative Adversarial Networks

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters
latent_dim = 100  # Dimension of the random noise input
batch_size = 64
lr = 0.0002
num_epochs = 50
img_dim = 28 * 28  # MNIST images are 28x28

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, img_dim),
            nn.Tanh()  # Scale output to [-1, 1]
        )
    
    def forward(self, z):
        return self.model(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output a probability
        )
    
    def forward(self, img):
        return self.model(img)

# Initialize models
generator = Generator(latent_dim)
discriminator = Discriminator()

# Loss and optimizers
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Prepare MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Scale images to [-1, 1]
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (real_imgs, _) in enumerate(train_loader):
        # Prepare real and fake data
        real_imgs = real_imgs.view(-1, img_dim).to(torch.float32)  # Flatten images
        real_labels = torch.ones((real_imgs.size(0), 1))  # Real labels = 1
        fake_labels = torch.zeros((real_imgs.size(0), 1))  # Fake labels = 0

        # Train Discriminator
        optimizer_D.zero_grad()
        z = torch.randn((real_imgs.size(0), latent_dim))  # Generate random noise
        fake_imgs = generator(z).detach()  # Generate fake images
        real_loss = criterion(discriminator(real_imgs), real_labels)
        fake_loss = criterion(discriminator(fake_imgs), fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn((real_imgs.size(0), latent_dim))  # Generate random noise
        fake_imgs = generator(z)  # Generate fake images
        g_loss = criterion(discriminator(fake_imgs), real_labels)  # Try to fool the discriminator
        g_loss.backward()
        optimizer_G.step()

        # Print progress
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(train_loader)} \
                  Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")

    # Save some generated images every epoch
    if epoch % 5 == 0:
        with torch.no_grad():
            z = torch.randn((16, latent_dim))  # Generate noise
            generated_imgs = generator(z).view(-1, 1, 28, 28)  # Reshape to image dimensions
            grid = torchvision.utils.make_grid(generated_imgs, normalize=True)
            torchvision.utils.save_image(grid, f'generated_epoch_{epoch}.png')

print("Training complete.")
