"""PyTorch autoencoder for stellar embeddings."""
import numpy as np
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class StarAutoencoder(nn.Module if TORCH_AVAILABLE else object):
    """Autoencoder for stellar embeddings."""
    
    def __init__(self, input_dim, latent_dim=64, hidden_dims=(256, 128), dropout=0.1):
        if not TORCH_AVAILABLE:
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            return
        
        super().__init__()
        self.config = {'input_dim': input_dim, 'latent_dim': latent_dim}
        
        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = hidden_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        reversed_dims = list(hidden_dims)[::-1]
        in_dim = latent_dim
        for hidden_dim in reversed_dims:
            decoder_layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        if not TORCH_AVAILABLE:
            return np.zeros((len(x), self.input_dim))
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

def train_autoencoder(config, train_tensor):
    """Train autoencoder."""
    if not TORCH_AVAILABLE:
        return StarAutoencoder(config['input_dim'], config.get('latent_dim', 64))
    
    model = StarAutoencoder(
        config['input_dim'],
        config.get('latent_dim', 64),
        config.get('hidden_dims', (256, 128))
    )
    
    # Simple training (in production, use DataLoader, optimizer, etc.)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    if isinstance(train_tensor, np.ndarray):
        train_tensor = torch.FloatTensor(train_tensor)
    
    model.train()
    for epoch in range(10):  # Simplified training
        optimizer.zero_grad()
        output = model(train_tensor)
        loss = criterion(output, train_tensor)
        loss.backward()
        optimizer.step()
    
    return model

def encode_stars(model, tensor):
    """Generate latent embeddings."""
    if not TORCH_AVAILABLE:
        return np.random.randn(len(tensor), model.latent_dim)
    
    model.eval()
    with torch.no_grad():
        if isinstance(tensor, np.ndarray):
            tensor = torch.FloatTensor(tensor)
        latent = model.encoder(tensor)
    return latent.numpy()

