import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


class FlowModelPipeline:
    """Pipeline for sampling and visualization with flow matching models."""
    
    def __init__(self, flow_model: torch.nn.Module, device=None):
        """
        Initialize the FlowModelPipeline.
        
        Args:
            flow_model: The trained flow model
            device: Device to run the model on (default: None, will use model's device)
        """
        self.flow_model = flow_model
        self.device = device if device is not None else next(flow_model.parameters()).device
        self.flow_model.to(self.device)
        self.flow_model.eval()
    
    @classmethod
    def from_pretrained(
        cls,
        model: torch.nn.Module,
        state_dict_path: str | None = None,
        device: torch.device | str | None = None,
    ):
        """
        Create a FlowModelPipeline from a pretrained model.

        Args:
            model: Instantiated model architecture
            state_dict_path: Optional path to a state_dict
            device: Target device
        """
        if state_dict_path is not None:
            state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(state_dict)

        if device is not None:
            model = model.to(device)

        model.eval()
        return cls(model, device=device)

        
    @torch.inference_mode()
    def sample(self, x_s, n_steps=100):
        """
        Sample from the flow model using Euler integration.
        
        Args:
            x_s: Source samples (starting points), shape [n_samples, dim]
            n_steps: Number of integration steps
            
        Returns:
            Final samples after flow integration
        """
        x = x_s.clone().to(self.device)
        dt = 1.0 / n_steps
        t_vals = torch.linspace(0, 1, steps=n_steps, device=self.device)
        
        for t in t_vals:
            t_batch = t * torch.ones(x.size(0), 1, device=self.device)
            v = self.flow_model(x, t_batch)
            x = x + v * dt
        
        return x
    
    @torch.inference_mode()
    def sample_with_history(self, x_s, n_steps=100):
        """
        Sample from the flow model and return the full trajectory.
        
        Args:
            x_s: Source samples (starting points), shape [n_samples, dim]
            n_steps: Number of integration steps
            
        Returns:
            A tuple (final_samples, history) where:
                - final_samples: Final samples after flow integration
                - history: List of samples at each time step
        """
        x = x_s.clone().to(self.device)
        dt = 1.0 / n_steps
        t_vals = torch.linspace(0, 1, steps=n_steps, device=self.device)
        
        history = [x.cpu().clone()]
        
        for t in t_vals:
            t_batch = t * torch.ones(x.size(0), 1, device=self.device)
            v = self.flow_model(x, t_batch)
            x = x + v * dt
            history.append(x.cpu().clone())
        
        return x, history
    
    @torch.inference_mode()
    def generate_animation(self, x_s, n_steps=100, figsize=(5, 5), 
                          xlim=(-6, 6), ylim=(-6, 6), interval=50, 
                          alpha=0.3, s=5, title="Flow Sampling"):
        """
        Generate an animation of the flow sampling process (for 2D data).
        
        Args:
            x_s: Source samples (starting points), shape [n_samples, 2]
            n_steps: Number of integration steps
            figsize: Figure size
            xlim: X-axis limits
            ylim: Y-axis limits
            interval: Delay between frames in milliseconds
            alpha: Scatter plot alpha value
            s: Scatter plot point size
            title: Animation title
            
        Returns:
            HTML animation object for display in Jupyter notebooks
        """
        if x_s.shape[1] != 2:
            raise ValueError("generate_animation only supports 2D data")
        
        x = x_s.clone().to(self.device)
        dt = 1.0 / n_steps
        t_vals = torch.linspace(0, 1, steps=n_steps, device=self.device)
        
        fig, ax = plt.subplots(figsize=figsize)
        scat = ax.scatter(x[:, 0].cpu().numpy(), x[:, 1].cpu().numpy(), 
                         s=s, alpha=alpha)
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        def update(frame):
            nonlocal x
            t = t_vals[frame]
            t_batch = t * torch.ones(x.size(0), 1, device=self.device)
            v = self.flow_model(x, t_batch)
            x = x + v * dt
            
            scat.set_offsets(x.cpu().numpy())
            ax.set_title(f"{title} - t = {t.item():.2f}")
            return scat,
        
        ani = FuncAnimation(fig, update, frames=n_steps, 
                          interval=interval, blit=True)
        
        plt.close(fig)
        return HTML(ani.to_jshtml())
    
    @torch.inference_mode()
    def visualize_trajectory(self, x_s, n_steps=100, figsize=(5, 5),
                            xlim=(-6, 6), ylim=(-6, 6), alpha=0.2,
                            n_trajectories=10):
        """
        Visualize sample trajectories (for 2D data).
        
        Args:
            x_s: Source samples, shape [n_samples, 2]
            n_steps: Number of integration steps
            figsize: Figure size
            xlim: X-axis limits
            ylim: Y-axis limits
            alpha: Line alpha value
            n_trajectories: Number of individual trajectories to plot
            
        Returns:
            matplotlib figure object
        """
        if x_s.shape[1] != 2:
            raise ValueError("visualize_trajectory only supports 2D data")
        
        _, history = self.sample_with_history(x_s, n_steps=n_steps)
        history = [h.numpy() for h in history]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot all points at start and end
        ax.scatter(history[0][:, 0], history[0][:, 1], 
                  c='blue', s=20, alpha=0.5, label='Start')
        ax.scatter(history[-1][:, 0], history[-1][:, 1], 
                  c='red', s=20, alpha=0.5, label='End')
        
        # Plot individual trajectories
        for i in range(min(n_trajectories, x_s.shape[0])):
            trajectory = torch.stack([h[i] for h in history])
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                   c='gray', alpha=alpha, linewidth=1)
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title("Flow Trajectories")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig
