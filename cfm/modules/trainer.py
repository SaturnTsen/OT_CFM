import torch
import torch.nn.functional as F
from tqdm import tqdm
from .pipeline import FlowModelPipeline
from .simple_flow import FlowModelBase
import ot

class Trainer:
    """Trainer class for flow matching models."""
    
    def __init__(self, 
                 flow_model: FlowModelBase,
                 dataloader: torch.utils.data.DataLoader,
                 optimizer: torch.optim.Optimizer,
                 n_epochs=100,
                 sigma=0.005,
                 coupling=None,
                 ):
        """
        Initialize the Trainer.
        
        Args:
            flow_model: The flow model to train (instance of FlowModelBase)
            dataloader: DataLoader providing (source, target) batches
            n_epochs: Number of training epochs
            lr: Learning rate
            sigma: Standard deviation for noise
            sample_from_coupling: Optional function to sample from coupling
            optimizer: Optimizer class to use (default: torch.optim.Adam)
        """
        self.flow_model = flow_model
        self.dataloader = dataloader
        self.n_epochs = n_epochs
        self.sigma = sigma
        self.coupling = coupling
        self.optimizer = optimizer
    
    def train(self, from_random_gaussian=False):
        """
        Train the flow model.
        
        Returns:
            The trained flow model
        """
        if from_random_gaussian:
            print("Training with random Gaussian noise as source samples.")
        else:
            print("Training with source samples from the dataset.")
            
        pbar = tqdm(range(self.n_epochs))
        device = next(self.flow_model.parameters()).device
        for epoch in pbar:
            total_loss = 0.0

            if hasattr(self.dataloader.dataset, "reshuffle"):
                # print("Reshuffling dataset for new epoch...")
                self.dataloader.dataset.reshuffle()

            for batch in self.dataloader:
                if from_random_gaussian:
                    x_s = torch.randn_like(batch[0]).to(device)
                    x_t = batch[0].to(device)
                else:
                    x_s = batch[0].to(device)
                    x_t = batch[1].to(device)
                
                if self.coupling is not None:
                    x_s, x_t = self.coupling(x_s, x_t)

                batch_size_curr = x_s.size(0)
                x_s = x_s.to(device)
                x_t = x_t.to(device)
                
                t = torch.rand(batch_size_curr, 1, device=device)
                mean = t * x_t + (1 - t) * x_s
                x = mean + torch.randn(batch_size_curr, 2, device=device) * self.sigma
                v = self.flow_model(x, t)
                
                v_target = x_t - x_s
                loss = F.mse_loss(v, v_target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch_size_curr
            
            pbar.set_description(
                f'Epoch [{epoch+1}/{self.n_epochs}], '
                f'Loss: {total_loss/len(self.dataloader.dataset):.4f}'
            )
        
        return self.flow_model