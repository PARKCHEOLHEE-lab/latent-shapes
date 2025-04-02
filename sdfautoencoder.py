import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim * 3
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 8),
            nn.ReLU(),
            nn.Linear(self.input_dim // 8, self.input_dim // 16),
            nn.ReLU(),
            nn.Linear(self.input_dim // 16, self.input_dim // 32),
            nn.ReLU(),
            nn.Linear(self.input_dim // 32, self.input_dim // 64),
            nn.ReLU(),
            nn.Linear(self.input_dim // 64, self.input_dim // 128),
            nn.ReLU(),
            nn.Linear(self.input_dim // 128, latent_dim),
            nn.Tanh()  # Latent vector 값을 [-1, 1]로 제한
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(Decoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Latent vector + 3D coordinates (x, y, z)
        self.decoder = nn.Sequential(
            nn.Linear(self.input_dim * 3 + self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 8),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 8, self.hidden_dim * 16),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 16, self.hidden_dim * 32),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 32, self.hidden_dim * 64),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 64, self.input_dim)
        )

    def forward(self, latent, xyz):
        
        sdf_ = self.decoder(torch.hstack([xyz, latent]))
        sdf_ = sdf_.reshape(xyz.shape[0], xyz.shape[1] // 3, -1)
        
        return sdf_


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(input_dim, latent_dim, hidden_dim)

    def forward(self, _xyz):
        xyz = _xyz.reshape(_xyz.shape[0], -1)
        latent = self.encoder(xyz)
        sdf_pred = self.decoder(latent, xyz)  # (batch, N, 1)
        return sdf_pred, latent
    

class SDFDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = [torch.tensor(np.load(data_path)) for data_path in self.data_path]

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        return self.data[idx]



if __name__ == "__main__":

    import os
    
    data_path = [f for f in os.listdir("./") if f.endswith(".npy")]
    data_path.sort()
    
    dataset = SDFDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=len(data_path), shuffle=True)
    
    autoencoder = Autoencoder(input_dim=32**3, latent_dim=3*8, hidden_dim=256)
    
    loss_function = nn.L1Loss()
    
    for i, data in enumerate(dataloader):
        
        xyz = data[:, :, :3].float()
        sdf = data[:, :, 3].unsqueeze(-1).float()
        
        sdf_, latent = autoencoder(xyz)

        assert sdf_.shape == sdf.shape
        
        print(data.shape)
        break

    pass
