"""
Physics-Informed Model (PhModel) for CO2 reduction predictions.
Combines neural networks with the complete electrochemical physics simulation.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import copy

# Import the complete physics engine
from models.gde_system import (
    System,
    diffusion_coefficients,
    salting_out_exponents,
    electrode_reaction_kinetics,
    electrode_reaction_potentials,
    chemical_reaction_rates
)


@dataclass
class PhysicsConfig:
    """Configuration for physics-informed model."""
    hidden_dim: int = 64
    dropout_rate: float = 0.1
    current_target: float = 200.0  # Target current density [A/m^2]
    grid_size: int = 1000
    voltage_bounds: Tuple[float, float] = (-1.25, 0.0)


class PhModel(nn.Module):
    """
    Physics-Informed Model for predicting Faradaic efficiencies.

    Architecture (EXACTLY matching the original):
    1. Neural network maps experimental inputs → latent physical parameters
    2. Complete gde_multi.System solves electrochemical physics
    3. Returns physics-based predictions for FE_CO and FE_C2H4

    This uses the full published physics engine with:
    - Complete mass transport equations
    - Butler-Volmer electrochemical kinetics
    - Carbonate equilibrium chemistry
    - Salting-out effects
    - Flow channel mass transfer
    """

    def __init__(self, config: PhysicsConfig = None, zlt_mu_stds: Optional[Tuple[float, float]] = None):
        super().__init__()
        self.config = config or PhysicsConfig()

        # Normalization parameters for Zero_eps_thickness (from normalized data)
        self.zlt_mu_stds = zlt_mu_stds or (0.0, 1.0)  # (mean, std) from normalized data

        # Neural network: 5 inputs → 6 latent physics parameters
        # EXACTLY matching the original architecture
        self.net = nn.Sequential(
            nn.Linear(5, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.hidden_dim, 6)  # 6 latent parameters
        )

        # Complete physics engine with learnable kinetic parameters
        erc = dict(electrode_reaction_kinetics)
        erc['i_0_CO'] = nn.parameter.Parameter(torch.tensor(erc['i_0_CO']))
        erc['i_0_C2H4'] = nn.parameter.Parameter(torch.tensor(erc['i_0_C2H4']))
        erc['i_0_H2b'] = nn.parameter.Parameter(torch.tensor(erc['i_0_H2b']))
        erc['alpha_CO'] = nn.parameter.Parameter(torch.tensor(erc['alpha_CO']))
        erc['alpha_C2H4'] = nn.parameter.Parameter(torch.tensor(erc['alpha_C2H4']))
        erc['alpha_H2b'] = nn.parameter.Parameter(torch.tensor(erc['alpha_H2b']))

        self.ph_model = System(
            diffusion_coefficients=diffusion_coefficients,
            salting_out_exponents=salting_out_exponents,
            electrode_reaction_kinetics=erc,
            electrode_reaction_potentials=electrode_reaction_potentials,
            chemical_reaction_rates=chemical_reaction_rates,
        )

        # Softmax for surface coverage fractions (must sum to ≤ 1)
        self.softmax = nn.Softmax(dim=-1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: experimental inputs → physics parameters → FE predictions.

        EXACTLY matches the original PhModel forward pass from the notebook.

        Args:
            x: Input tensor (batch_size, 5) with columns (NORMALIZED):
               [AgCu Ratio, Naf vol (ul), Sust vol (ul), Zero_eps_thickness, Catalyst mass loading]

        Returns:
            Predictions tensor (batch_size, 2) with [FE_C2H4, FE_CO]
        """
        # Validate input
        assert x.ndim == 2 and x.shape[1] == 5, f"Expected input shape (N, 5), got {x.shape}"

        # Neural network maps inputs to latent parameters
        latents = self.net(x)  # (batch_size, 6)

        # Extract and transform latent parameters to physical parameters
        # EXACTLY matching original transformations:

        # 1. Pore radius (log-normal distribution)
        r = 40e-9 * torch.exp(latents[..., 0:1])  # [m]

        # 2. Porosity (sigmoid to ensure 0 < eps < 1)
        eps = torch.sigmoid(latents[..., 1:2])  # dimensionless

        # 3. Zero layer thickness (denormalized from NORMALIZED input)
        # This is the key insight: x[..., 3] is NORMALIZED Zero_eps_thickness
        zlt = (x[..., 3:4] * self.zlt_mu_stds[1] + self.zlt_mu_stds[0])  # [m]

        # 4. Layer thickness
        L = zlt / (1 - eps)  # [m] - original had no +1e-8 protection

        # 5. Mass transfer coefficient factor
        K_dl_factor = torch.exp(latents[..., 2:3])

        # 6. Surface coverage fractions (softmax, scaled by 2 for gradients)
        theta_logits = 2 * latents[..., 3:6]  # Scale for better gradients
        thetas_vec = self.softmax(theta_logits)  # (batch_size, 3)

        # Convert to dictionary format expected by physics engine
        thetas = {
            'CO': thetas_vec[..., 0:1],
            'C2H4': thetas_vec[..., 1:2],
            'H2b': thetas_vec[..., 2:3]
        }

        # Calculate gas diffusion layer mass transfer coefficient
        gdl_mass_transfer_coefficient = K_dl_factor * self.ph_model.bruggeman(
            self.ph_model.diffusion_coefficients['CO2'], eps) / r

        # Solve complete physics equations using the full System
        solution = self.ph_model.solve_current(
            i_target=self.config.current_target,
            eps=eps,
            r=r,
            L=L,
            thetas=thetas,
            gdl_mass_transfer_coeff=gdl_mass_transfer_coefficient,
            grid_size=self.config.grid_size,
            voltage_bounds=self.config.voltage_bounds
        )

        # Return FE predictions: [FE_C2H4, FE_CO] - EXACTLY matching original order
        output = torch.cat([solution['fe_c2h4'], solution['fe_co']], dim=-1)

        return output

    def get_physics_parameters(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get the intermediate physics parameters for interpretation.

        Args:
            x: Input tensor (batch_size, 5) - normalized features

        Returns:
            Dictionary of physics parameters
        """
        with torch.no_grad():
            latents = self.net(x)

            r = 40e-9 * torch.exp(latents[..., 0:1])
            eps = torch.sigmoid(latents[..., 1:2])
            zlt = (x[..., 3:4] * self.zlt_mu_stds[1] + self.zlt_mu_stds[0])
            L = zlt / (1 - eps)
            K_dl_factor = torch.exp(latents[..., 2:3])

            theta_logits = 2 * latents[..., 3:6]
            thetas_vec = self.softmax(theta_logits)

            return {
                'pore_radius': r,
                'porosity': eps,
                'zero_layer_thickness': zlt,
                'layer_thickness': L,
                'mass_transfer_factor': K_dl_factor,
                'theta_CO': thetas_vec[..., 0:1],
                'theta_C2H4': thetas_vec[..., 1:2],
                'theta_H2b': thetas_vec[..., 2:3]
            }
