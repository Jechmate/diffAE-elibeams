import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from src.utils import ExperimentDataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb

def get_data(batch_size, seed=42):
    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    # Define transforms
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5)
    ])
    
    # Load the dataset
    dataset = ExperimentDataset('data/params.csv', 'data/with_gain', transform=transforms, features=["E","P","ms"], exclude=[])
    
    # Split dataset into training and validation
    val_percent = 0.2
    val_size = int(len(dataset) * val_percent)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
            # nn.Dropout(p=0.2),
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
            # nn.Dropout(p=0.2),
        )

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder: Input size [B, 1, 256, 512]
        self.encoder = nn.Sequential(
            DoubleConv(1, 16),
            Down(16, 32),
            Down(32, 64),
            Down(64, 128),
            Down(128, 128),
            nn.ReLU()
        )
        
        # Flatten
        self.fc1 = nn.Linear(128 * 16 * 32, 256)  # Flattened size
        self.fc_mu = nn.Linear(256, latent_dim)  # Mean vector
        self.fc_logvar = nn.Linear(256, latent_dim)  # Log variance vector
        
        # Decoder
        self.fc2 = nn.Linear(latent_dim, 256)
        self.fc3 = nn.Linear(256, 128 * 16 * 32)
        
        self.decoder = nn.Sequential(
            Up(128, 128),
            Up(128, 64),
            Up(64, 32),
            Up(32, 16),
            DoubleConv(16, 1),
            nn.Tanh()
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
        h = F.relu(self.fc3(h)).view(-1, 128, 16, 32)  # Unflatten and reshape
        h = self.decoder(h)
        return h

    def forward(self, x):
        """Defines the forward pass of the VAE."""
        mu, logvar = self.encode(x)  # Encode input to latent distribution
        z = self.reparameterize(mu, logvar)  # Reparameterization trick
        x_recon = self.decode(z)  # Decode latent vector back to image space
        return x_recon, mu, logvar


class SimpleVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super(SimpleVAE, self).__init__()
        
        self.latent_dim = latent_dim
        # Input images are of size [B, 1, 256, 512]
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # [B, 32, 128, 256]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 64, 128]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # [B, 128, 32, 64]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# [B, 256, 16, 32]
            nn.ReLU()
        )
        
        # Adjusted flatten dimension
        self.flatten_dim = 256 * 16 * 32
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # [B, 128, 32, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 64, 128]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # [B, 32, 128, 256]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),    # [B, 1, 256, 512]
            nn.Tanh()  # Output values between -1 and 1
        )
        
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(-1, self.flatten_dim)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 256, 16, 32)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


# Loss function (combines reconstruction loss and KL divergence)
def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Total loss
    total_loss = recon_loss + kld
    return total_loss, recon_loss, kld

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 2000
    batch_size = 64
    wandb.login(relogin=True, force=True, key="6418556c25c3d904e43ecd582234a0336a5de781")
    # Initialize wandb
    wandb.init(project="vae", name="simplified_vae_128lat", config={
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": 1e-3,
        "latent_dim": 64,
    })
    
    # Instantiate model and optimizer
    vae = SimpleVAE(latent_dim=wandb.config.latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=wandb.config.learning_rate)
    train_loader, val_loader = get_data(wandb.config.batch_size)
    
    # Training loop
    for epoch in tqdm(range(wandb.config.epochs)):
        vae.train()
        train_loss = 0
        total_recon_loss = 0
        total_kld_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['image'].to(device)
            optimizer.zero_grad()
            
            # Forward pass through VAE
            recon_batch, mu, logvar = vae(inputs)
            logvar = torch.clamp(logvar, min=-100, max=100)
            
            # Compute loss with KL annealing
            kld_weight = min(1.0, epoch / (num_epochs / 2))  # Annealing factor reaches 1 halfway through training
            loss, recon_loss, kld_loss = loss_function(recon_batch, inputs, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate losses
            train_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kld_loss += kld_loss.item()
            
            # Log batch-wise losses to wandb
            wandb.log({
                "batch_loss": loss.item() / inputs.size(0),
                "batch_recon_loss": recon_loss.item() / inputs.size(0),
                "batch_kld_loss": kld_loss.item() / inputs.size(0),
            })
        
        # Compute average losses per epoch
        avg_loss = train_loss / len(train_loader.dataset)
        avg_recon_loss = total_recon_loss / len(train_loader.dataset)
        avg_kld_loss = total_kld_loss / len(train_loader.dataset)
        
        # Log epoch-wise losses to wandb
        wandb.log({
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "avg_recon_loss": avg_recon_loss,
            "avg_kld_loss": avg_kld_loss
        })
        
        # Validation
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            vae.eval()
            val_loss = 0
            val_recon_loss = 0
            val_kld_loss = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    inputs = batch['image'].to(device)
                    recon_batch, mu, logvar = vae(inputs)
                    loss, recon_loss, kld_loss = loss_function(recon_batch, inputs, mu, logvar)
                    val_loss += loss.item()
                    val_recon_loss += recon_loss.item()
                    val_kld_loss += kld_loss.item()
                
                # Compute average validation losses
                avg_val_loss = val_loss / len(val_loader.dataset)
                avg_val_recon_loss = val_recon_loss / len(val_loader.dataset)
                avg_val_kld_loss = val_kld_loss / len(val_loader.dataset)
                
                # Log validation losses
                wandb.log({
                    "epoch": epoch + 1,
                    "val_avg_loss": avg_val_loss,
                    "val_avg_recon_loss": avg_val_recon_loss,
                    "val_avg_kld_loss": avg_val_kld_loss
                })
                
                # Visualize reconstructions on validation data
                # Get a batch from validation data
                val_batch = next(iter(val_loader))
                inputs = val_batch['image'].to(device)
                recon_batch, _, _ = vae(inputs)
                sample = recon_batch.cpu()
                inputs_cpu = inputs.cpu()
                # Log the first 8 original and reconstructed images
                comparison = torch.cat([inputs_cpu[:8], sample[:8]])
                grid = torchvision.utils.make_grid(comparison, nrow=8, normalize=True, range=(-1, 1))
                wandb.log({"val_reconstructions": [wandb.Image(grid, caption="Top: Originals, Bottom: Reconstructions")]})
                
            # Save model checkpoint
            torch.save(vae.state_dict(), f'models/vae_epoch_{epoch}.pth')
            print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, KLD Loss: {avg_kld_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    # Save the final model
    torch.save(vae.state_dict(), 'models/vae_final.pth')
    wandb.finish()

if __name__ == "__main__":
    train()
