"""Finance autoencoder."""
import numpy as np
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class FinanceAutoencoder(nn.Module if TORCH_AVAILABLE else object):
    """Autoencoder for finance embeddings."""
    
    def __init__(self, input_dim, latent_dim=32, hidden_dims=(128, 64)):
        if not TORCH_AVAILABLE:
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            return
        
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim)
        )
    
    def forward(self, x):
        if not TORCH_AVAILABLE:
            return np.zeros((len(x), self.input_dim))
        latent = self.encoder(x)
        return self.decoder(latent)

def train_finance_autoencoder(config, train_tensor):
    """Train finance autoencoder."""
    if not TORCH_AVAILABLE:
        return FinanceAutoencoder(config['input_dim'])
    
    model = FinanceAutoencoder(
        config['input_dim'],
        config.get('latent_dim', 32),
        config.get('hidden_dims', (128, 64))
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    if isinstance(train_tensor, np.ndarray):
        train_tensor = torch.FloatTensor(train_tensor)
    
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(train_tensor)
        loss = criterion(output, train_tensor)
        loss.backward()
        optimizer.step()
    
    return model

