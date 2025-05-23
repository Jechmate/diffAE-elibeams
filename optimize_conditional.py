import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import necessary modules from the project
from src.diffusion_differentiable import SpacedDiffusion
from src.modules import UNet_conditional
from metrics import create_sections_list, cosine_step_schedule
from src.utils import deflection_biexp_calc, calc_spec

class SpectrumWeightedMeanOptimizer:
    def __init__(
        self,
        model_path,
        device="cuda",
        pressure=15.0,
        laser_energy=25.0,
        acquisition_time_ms=20.0,
        target_mean_energy=25.0,  # Target mean energy to optimize for
        optimization_steps=100,
        lr=0.01,
        weight_function="gaussian",
        sigma=5.0,  # Standard deviation for Gaussian weight function
        batch_size=4
    ):
        self.device = device
        self.pressure = pressure
        self.laser_energy = laser_energy
        self.acquisition_time_ms = acquisition_time_ms
        self.target_mean_energy = target_mean_energy
        self.optimization_steps = optimization_steps
        self.lr = lr
        self.weight_function = weight_function
        self.sigma = sigma
        self.batch_size = batch_size
        # Initialize model
        self.model = UNet_conditional(img_width=128, img_height=64, feat_num=3, device=device).to(device)
        self.load_model(model_path)
        
        # Initialize sampler
        self.sampler = SpacedDiffusion(
            beta_start=1e-4,
            beta_end=0.02,
            noise_steps=1000,
            section_counts=create_sections_list(6, 6, cosine_step_schedule),
            img_height=64,
            img_width=128,
            device=device,
            rescale_timesteps=False
        )
        
        # Calculate deflection once (since it doesn't depend on parameters we optimize)
        self.deflection_MeV, self.deflection_MeV_dx = deflection_biexp_calc(
            batch_size=1,
            hor_image_size=128,
            electron_pointing_pixel=62
        )
        self.deflection_MeV = self.deflection_MeV.to(device)
        self.deflection_MeV_dx = self.deflection_MeV_dx.to(device)
        
        # Initialize optimization history
        self.energy_history = []
        self.loss_history = []
        self.spectrum_history = []
        # Add parameter tracking
        self.laser_energy_history = []
        self.pressure_history = []
        self.acquisition_time_history = []
        
    def load_model(self, model_path):
        """Load the pre-trained model"""
        try:
            ckpt = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(ckpt)
            self.model.eval()  # Set to evaluation mode
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def calculate_weighted_mean(self, spectrum, energy_values):
        """
        Calculate the weighted mean of the spectrum where each intensity is weighted by its energy.
        
        Args:
            spectrum (torch.Tensor): The intensity values of the spectrum
            energy_values (torch.Tensor): The corresponding energy values
            
        Returns:
            tuple: (weighted_mean, weights) where weighted_mean is the calculated mean and weights are the applied weights
        """
        # Ensure inputs are on the correct device
        spectrum = spectrum.to(self.device)
        energy_values = energy_values.to(self.device)
       
        weights = torch.ones_like(energy_values, requires_grad=True)
        
        # Normalize weights to sum to 1
        weights = weights / weights.sum()
        
        # Calculate weighted mean: sum(weights * spectrum * energy_values) / sum(weights * spectrum)
        weighted_mean = torch.sum(weights * spectrum * energy_values) / torch.sum(weights * spectrum)
        return weighted_mean, weights

    def optimize(self):
        """Run the optimization process to find optimal energy parameter"""
        # Initialize energy parameter with requires_grad=True for gradient computation
        
        y = torch.cat([
            torch.tensor([self.laser_energy], device=self.device), 
            torch.tensor([self.pressure], device=self.device), 
            torch.tensor([self.acquisition_time_ms], device=self.device)
        ]).unsqueeze(0)
        y.requires_grad = True

        # Create optimizer
        optimizer = optim.Adam([y], lr=self.lr)
        
        # Track best result
        best_energy = 0
        best_loss = float('inf')
        
        # Run optimization
        for step in tqdm(range(self.optimization_steps), desc="Optimizing energy"):
            optimizer.zero_grad()
            
            x = self.sampler.ddim_sample_loop(model=self.model, y=y, cfg_scale=1, device=self.device, eta=1, n=self.batch_size).type(torch.float32)
            deflection_MeV, _ = deflection_biexp_calc(self.batch_size, 128, 64, 0.137)
            deflection_MeV = deflection_MeV.to(self.device)
            _, x_spectr = calc_spec(x.unsqueeze(1).to(self.device)/255, 
                                        62,
                                        deflection_MeV, 
                                        acquisition_time_ms=torch.tensor([self.acquisition_time_ms], device=self.device).repeat(self.batch_size), 
                                        resize=(64, 128),
                                        image_gain=0,
                                        device=self.device,
                                        deflection_MeV_dx=None)
            weighted_mean, weights = self.calculate_weighted_mean(x_spectr, deflection_MeV)
            loss = (weighted_mean - self.target_mean_energy) ** 2
            
            # Print detailed debug information
            if step % 10 == 0:
                print(f"Step {step} details:")
                print(f"  Weighted mean: {weighted_mean.item():.4f}")
                print(f"  Target mean: {self.target_mean_energy:.4f}")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Grad enabled: {torch.is_grad_enabled()}")
            
            # Backward pass
            loss.backward()
            
            # if step % 10 == 0:
            #     if weighted_mean.grad is not None:
            #         print(f"  Spectrum energy gradient: {weighted_mean.grad.item():.6f}")
            #     else:
            #         print("  Spectrum energy gradient is None!")
            
            # Update parameters
            optimizer.step()
            
            # Store history for visualization
            self.energy_history.append(weighted_mean.item())
            self.loss_history.append(loss.item())
            
            # Track the current parameter values
            current_laser_energy = y[0, 0].item()
            current_pressure = y[0, 1].item()
            current_acquisition_time = y[0, 2].item()
            
            self.laser_energy_history.append(current_laser_energy)
            self.pressure_history.append(current_pressure)
            self.acquisition_time_history.append(current_acquisition_time)
            
            # Save spectrum periodically
            if step % 10 == 0 or step == self.optimization_steps - 1:
                self.spectrum_history.append({
                    'spectrum_energy': weighted_mean.item(),
                    'spectrum': x_spectr[0].detach().cpu().numpy(),
                    'energy_values': deflection_MeV[0].detach().cpu().numpy(),
                    'weights': weights.detach().cpu().numpy(),
                    'laser_energy': current_laser_energy,
                    'pressure': current_pressure,
                    'acquisition_time': current_acquisition_time
                })
            
            # Track best result
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_energy = weighted_mean.item()
                
            # Print progress
            if step % 10 == 0:
                print(f"Step {step}: Spectrum energy = {weighted_mean.item():.2f}, Loss = {loss.item():.4f}")
            
        print(f"Optimization completed. Best energy: {best_energy:.2f} with loss: {best_loss:.4f}")
        return best_energy, best_loss

    def plot_results(self):
        """Visualize the optimization results with multiple plots"""
        fig, axs = plt.subplots(3, 2, figsize=(14, 15))
        
        # Plot 1: Spectrum Energy history
        axs[0, 0].plot(self.energy_history)
        axs[0, 0].set_title('Spectrum Energy vs. Iterations')
        axs[0, 0].set_xlabel('Iteration')
        axs[0, 0].set_ylabel('Spectrum Energy (MeV)')
        axs[0, 0].axhline(y=self.target_mean_energy, color='r', linestyle='--', label=f'Target: {self.target_mean_energy} MeV')
        axs[0, 0].grid(True)
        axs[0, 0].legend()
        
        # Plot 2: Loss history
        axs[0, 1].plot(self.loss_history)
        axs[0, 1].set_title('Loss vs. Iterations')
        axs[0, 1].set_xlabel('Iteration')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].set_yscale('log')
        axs[0, 1].grid(True)
        
        # Plot 3: Laser Energy parameter change
        axs[1, 0].plot(self.laser_energy_history)
        axs[1, 0].set_title('Laser Energy Parameter vs. Iterations')
        axs[1, 0].set_xlabel('Iteration')
        axs[1, 0].set_ylabel('Laser Energy')
        axs[1, 0].axhline(y=self.laser_energy, color='r', linestyle='--', label=f'Initial: {self.laser_energy}')
        axs[1, 0].grid(True)
        axs[1, 0].legend()
        
        # Plot 4: Pressure parameter change
        axs[1, 1].plot(self.pressure_history)
        axs[1, 1].set_title('Pressure Parameter vs. Iterations')
        axs[1, 1].set_xlabel('Iteration')
        axs[1, 1].set_ylabel('Pressure')
        axs[1, 1].axhline(y=self.pressure, color='r', linestyle='--', label=f'Initial: {self.pressure}')
        axs[1, 1].grid(True)
        axs[1, 1].legend()
        
        # Plot 5: Acquisition Time parameter change
        axs[2, 0].plot(self.acquisition_time_history)
        axs[2, 0].set_title('Acquisition Time Parameter vs. Iterations')
        axs[2, 0].set_xlabel('Iteration')
        axs[2, 0].set_ylabel('Acquisition Time (ms)')
        axs[2, 0].axhline(y=self.acquisition_time_ms, color='r', linestyle='--', label=f'Initial: {self.acquisition_time_ms}')
        axs[2, 0].grid(True)
        axs[2, 0].legend()
        
        # Plot 6: Selected spectra at different iterations
        if len(self.spectrum_history) > 0:
            # Select up to 5 spectra to plot
            indices = np.linspace(0, len(self.spectrum_history)-1, min(5, len(self.spectrum_history))).astype(int)
            
            for idx in indices:
                spec_data = self.spectrum_history[idx]
                step_idx = idx * 10 if idx < len(self.spectrum_history) - 1 else self.optimization_steps - 1
                axs[2, 1].plot(
                    spec_data['energy_values'], 
                    spec_data['spectrum'], 
                    label=f'Step {step_idx}, E={spec_data["spectrum_energy"]:.2f} MeV'
                )
            
            axs[2, 1].set_title('Spectrum Evolution')
            axs[2, 1].set_xlabel('Energy (MeV)')
            axs[2, 1].set_ylabel('Intensity')
            axs[2, 1].grid(True)
            axs[2, 1].legend()
        
        plt.tight_layout()
        plt.savefig('optimization_results.png')
        plt.show()
        
        # Create a separate figure for the final spectrum with weights
        if len(self.spectrum_history) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            final_spec = self.spectrum_history[-1]
            ax.plot(final_spec['energy_values'], final_spec['spectrum'], 'b-', label='Final Spectrum')
            ax.set_xlabel('Energy (MeV)')
            ax.set_ylabel('Intensity', color='b')
            ax.tick_params(axis='y', labelcolor='b')
            ax.grid(True)
            
            # Add second y-axis for weights
            ax2 = ax.twinx()
            # Handle the case where weights have batch dimension
            if len(final_spec['weights'].shape) > 1:
                weights_to_plot = np.mean(final_spec['weights'], axis=0)
            else:
                weights_to_plot = final_spec['weights']
            ax2.plot(final_spec['energy_values'], weights_to_plot, 'r--', label='Weights')
            ax2.set_ylabel('Weight', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            title = f'Final Spectrum (Energy = {final_spec["spectrum_energy"]:.2f} MeV)\n'
            title += f'Params: Laser={final_spec["laser_energy"]:.2f}, Pressure={final_spec["pressure"]:.2f}, Time={final_spec["acquisition_time"]:.2f}ms'
            ax.set_title(title)
            
            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2)
            
            plt.tight_layout()
            plt.savefig('final_spectrum.png')
            plt.show()

def main():
    # Set parameters
    model_path = "models/cossched_full/ema_ckpt.pt"  # Adjust to your model path
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    
    # Create optimizer
    optimizer = SpectrumWeightedMeanOptimizer(
        model_path=model_path,
        device=device,
        pressure=15.0,
        acquisition_time_ms=20.0,
        laser_energy=25.0,
        target_mean_energy=40.0,  # Target mean energy in MeV
        optimization_steps=100,
        lr=2,
        weight_function="gaussian",
        sigma=3.0,
        batch_size=2
    )
    
    # Run optimization
    best_energy, best_loss = optimizer.optimize()
    
    # Plot results
    optimizer.plot_results()
    
    print(f"Optimization complete. Best energy parameter: {best_energy:.2f}")

if __name__ == "__main__":
    main()
