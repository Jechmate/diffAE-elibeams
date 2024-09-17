import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VAE(nn.Module):
    def __init__(self, img_channels=1, latent_dim=256):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder: Input size [B, 1, 256, 512]
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 128, 256]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 64, 128]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 32, 64]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 16, 32]
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, 8, 16]
            nn.ReLU(),
        )
        
        # Flatten
        self.fc1 = nn.Linear(512 * 8 * 16, 1024)  # Flattened size is 512 * 8 * 16
        self.fc_mu = nn.Linear(1024, latent_dim)  # Mean vector
        self.fc_logvar = nn.Linear(1024, latent_dim)  # Log variance vector
        
        # Decoder
        self.fc2 = nn.Linear(latent_dim, 1024)
        self.fc3 = nn.Linear(1024, 512 * 8 * 16)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 16, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 32, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 64, 128]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 128, 256]
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),  # [B, 1, 256, 512]
            nn.Tanh()  # Use sigmoid to constrain output between [0, 1]
        )
        
    def encode(self, x):
        """Encodes the input image into a latent distribution (mean and log variance)."""
        h = self.encoder(x)  # Convolutional layers
        h = h.view(h.size(0), -1)  # Flatten
        h = F.relu(self.fc1(h))  # Fully connected layer
        mu = self.fc_mu(h)  # Mean of the latent space
        logvar = self.fc_logvar(h)  # Log variance of the latent space
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Samples from the latent space using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)  # Random noise
        return mu + eps * std  # Reparameterized latent vector

    def decode(self, z):
        """Decodes the latent vector back into the image space."""
        h = F.relu(self.fc2(z))  # Fully connected layer
        h = F.relu(self.fc3(h)).view(-1, 512, 8, 16)  # Unflatten and reshape
        return self.decoder(h)

    def forward(self, x):
        """Defines the forward pass of the VAE."""
        mu, logvar = self.encode(x)  # Encode input to latent distribution
        z = self.reparameterize(mu, logvar)  # Reparameterization trick
        x_recon = self.decode(z)  # Decode latent vector back to image space
        return x_recon, mu, logvar


# Loss function (combines reconstruction loss and KL divergence)
def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')  # Reconstruction loss (MSE)
    
    # KL divergence term
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return MSE + KLD


device = "cuda:1"
num_epochs = 100
# Instantiate model and optimizer
vae = VAE().to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Example training loop
for epoch in range(num_epochs):
    vae.train()
    train_loss = 0
    for batch in train_loader:
        inputs = batch['image'].to(device)
        # inputs = batch.to(device)  # Input images [B, 1, 256, 512]
        optimizer.zero_grad()
        
        # Forward pass through VAE
        recon_batch, mu, logvar = vae(inputs)
        
        # Compute loss
        loss = loss_function(recon_batch, inputs, mu, logvar)
        loss.backward()
        
        # Backpropagation and optimization
        train_loss += loss.item()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}')