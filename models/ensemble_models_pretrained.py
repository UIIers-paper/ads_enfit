import torch
import torch.nn as nn

class DiffusionRegressor(nn.Module):
    def __init__(self, diffusion_model, regressor):
        super().__init__()
        self.diffusion = diffusion_model  # Mô hình diffusion (đã huấn luyện)
        self.regressor = regressor        # Mạng hồi quy (MLP/Transformer)

    def forward(self, x):
        z = self.diffusion.encode(x)     # Trích xuất latent code
        return self.regressor(z)
    


class VAERegressor(nn.Module):
    def __init__(self, encoder, decoder, regressor):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.regressor = regressor

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.regressor(z), mu, logvar
    



class GraphRegressor(nn.Module):
    def __init__(self, gnn, regressor):
        super().__init__()
        self.gnn = gnn
        self.regressor = regressor

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        graph_emb = self.gnn(x, edge_index)
        return self.regressor(graph_emb)
    



class FusionRegressor(nn.Module):
    def __init__(self, diffusion_model, vae_model, gnn_model, regressor):
        super().__init__()
        self.diffusion = diffusion_model
        self.vae = vae_model
        self.gnn = gnn_model
        self.regressor = regressor

    def forward(self, x, graph_data):
        z_diff = self.diffusion.encode(x)
        mu, logvar = self.vae.encoder(x)
        z_vae = self.vae.reparameterize(mu, logvar)
        z_gnn = self.gnn(graph_data)
        combined = torch.cat([z_diff, z_vae, z_gnn], dim=1)
        return self.regressor(combined)