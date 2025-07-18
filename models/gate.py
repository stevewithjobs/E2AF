from typing import Optional, Sequence
import functools
import operator
import collections
import math

import torch

from models.smoe_config import SpatialMoEConfig
from utils.noise import NormalNoise

from typing import List, Tuple


class SpatialGate2d(torch.nn.Module):
    """Base class for gating layers."""

    def __init__(self, smoe_config: SpatialMoEConfig) -> None:
        """
        smoe_config is the configuration describing the SMoE.

        This outputs tensors of size [batch, num_experts, *], which
        can just be multiplied to do expert selection.

        """
        super().__init__()
        self.smoe_config = smoe_config
        if smoe_config.gate_mask is not None:
            smoe_config.in_planes += 1
        if smoe_config.out_planes > smoe_config.num_experts:
            raise ValueError(f'SpatialConvGate: out planes {smoe_config.out_planes}'
                             f' > num experts {smoe_config.num_experts}')
        self.noise_std = smoe_config.noise_std_scale / smoe_config.num_experts if smoe_config.noise else None
        self.noise = NormalNoise(0, self.noise_std) if smoe_config.noise else None
        if smoe_config.load_weight and not smoe_config.noise:
            smoe_config.load_weight = 0.0  # Disable silently for now.
        # Assigned by child classes.
        self.gate = lambda _: None
        self.gate_act = smoe_config.gate_act() if smoe_config.gate_act is not None else None

    @staticmethod
    def importance_loss(routing_weights: torch.Tensor) -> torch.Tensor:
        """Compute the importance auxiliary loss from routing_weights.

        The importance of an expert is just the sum of its routing weights.

        The loss is the coefficient of variation (squared) over the
        expert importance.

        """
        # The routing weights are (batch, expert, spatial...), so to
        # compute the importance of each expert, we sum out the other dims.
        expert_importance = routing_weights.sum(
            dim=(0,) + tuple(range(routing_weights.ndim))[2:])
        imp_std = expert_importance.std()
        imp_mean = expert_importance.mean()
        return (imp_std / imp_mean)**2

    def load_loss(self, routing_weights: torch.Tensor,
                  noisy_routing_weights: torch.Tensor) -> torch.Tensor:
        """Compute the load auxiliary loss from routing_weights.

        As a differentiable proxy for the load (which is discrete), we
        look at the probability that an expert is selected if we
        resample the noise.

        The loss is the coefficient of variation (squared) of this.

        """
        # Get the smallest routing weight of a selected expert at each point.
        threshold = noisy_routing_weights.topk(
            k=self.smoe_config.out_planes, dim=1, sorted=True).values[:, -1, :]
        # Find how far each routing weight is from this threshold.
        distance_to_selection = threshold.unsqueeze(1) - routing_weights
        # Compute the probability that, if we resampled the noise, an
        # expert would be selected.
        try:
            p = 1.0 - torch.distributions.normal.Normal(
                0.0, self.noise_std).cdf(distance_to_selection)
        except ValueError as e:
            breakpoint()
            raise
        # Compute the load over all samples.
        load = p.sum(dim=0)
        load_std = load.std()
        load_mean = load.mean()
        return (load_std / load_mean)**2

    def spatial_agreement_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Compute the spatial agreement auxiliary loss from routing_weights.

        This loss encourages the routing weights to take similar values
        at each spatial location across a batch.

        To encourage similar values, we minimize the standard deviation
        of the routing weights of each expert at each point. The loss
        is then the average stdev across spatial locations for each
        expert, and then this average summed over experts.

        Note: Coefficient of variation was considered instead of plain
        standard deviation, but the mean is often small or 0, making
        it behave poorly.

        """
        expert_spatial_std = routing_weights.std(dim=0)
        mean_expert_std = expert_spatial_std.mean(
            dim=tuple(range(expert_spatial_std.ndim))[1:])
        spatial_agreement = mean_expert_std.sum()
        return spatial_agreement

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Possibly concatenate the gate mask.
        if self.smoe_config.gate_mask is not None:
            self.smoe_config.gate_mask = self.smoe_config.gate_mask.to(dtype=x.dtype, device=x.device)
            expanded_mask = self.smoe_config.gate_mask.repeat(x.size(0), 1, 1, 1)
            x = torch.cat((x, expanded_mask), dim=1)
        # Use the gate to compute the actual routing weights.
        routing_weights = self.gate(x)
        # Apply activation, if present.
        if self.gate_act is not None:
            routing_weights = self.gate_act(routing_weights)
        if routing_weights is None:
            raise RuntimeError('No routing weights')
        # Compute auxiliary losses.
        self.aux_losses: dict[str, Optional[torch.Tensor]] = {
            'importance_loss': None,
            'spatial_agreement_loss': None,
            'load_loss': None
        }
        if self.training and self.smoe_config.importance_weight:
            if self.smoe_config.norm_weighted:
                # Need to apply the softmax here when computing importance.
                self.aux_losses['importance_loss'] = self.smoe_config.importance_weight * self.importance_loss(
                    routing_weights.softmax(dim=1))
            else:
                self.aux_losses['importance_loss'] = self.smoe_config.importance_weight * self.importance_loss(
                    routing_weights)
        if self.training and self.smoe_config.spatial_agreement_weight:
            self.aux_losses['spatial_agreement_loss'] = self.smoe_config.spatial_agreement_weight * self.spatial_agreement_loss(
                routing_weights)
        # Only add noise when training.
        if self.noise and self.training:
            noisy_routing_weights = routing_weights + self.noise.get_noise(
                routing_weights)
            if self.smoe_config.load_weight:
                self.aux_losses['load_loss'] = self.smoe_config.load_weight * self.load_loss(
                    routing_weights, noisy_routing_weights)
            routing_weights = noisy_routing_weights
        # Apply absolute value if needed.
        if self.smoe_config.absval_routing:
            routing_weights = routing_weights.abs()
        # Normalize with softmax.
        if self.smoe_config.norm_weighted:
            routing_weights = routing_weights.softmax(dim=1)
        return routing_weights


class SpatialLinearGate2d(SpatialGate2d):
    """Linear gating function."""

    def __init__(self, smoe_config: SpatialMoEConfig,
                 node_num: int) -> None:
        super().__init__(smoe_config)
        input_shape = (smoe_config.in_planes, node_num)  # Add channel dim.
        in_neurons = functools.reduce(operator.mul, input_shape, 1)
        out_shape = (smoe_config.num_experts,) + input_shape[1:]
        out_neurons = functools.reduce(operator.mul, out_shape, 1)
        layers = [('flatten', torch.nn.Flatten(start_dim=2)),
                  ('linear', torch.nn.Linear(in_neurons, out_neurons, bias=False)),
                  ('unflatten', torch.nn.Unflatten(-1, out_shape))]
        self.gate = torch.nn.Sequential(collections.OrderedDict(layers))


class SpatialLatentTensorGate2d(SpatialGate2d):
    """A gating function that learns a latent tensor for routing.

    This tensor does not depend on the input data and is directly
    optimized.

    """

    def __init__(self, smoe_config: SpatialMoEConfig,
                 node_num: int) -> None:
        super().__init__(smoe_config)
        self.latent_routing = torch.nn.Parameter(torch.empty(  # pyright: ignore reportPrivateImportUsage
            node_num, smoe_config.num_experts))
        self.reset_parameters()

        def gate_func(x: torch.Tensor) -> torch.Tensor:
            return self.latent_routing.repeat(x.size(0), x.size(1), 1, 1)

        self.gate = gate_func

    def reset_parameters(self):
        #torch.nn.init.normal_(self.latent_routing, 0, 1.0 / self.smoe_config.num_experts)
        # Currently assuming gain is 1 (which holds for linear, identity,
        # and sigmoid activations).
        # This essentially adapts Kaiming uniform initialization.
        gain = 1.0
        fan = self.smoe_config.out_planes / self.smoe_config.num_experts
        bound = gain * math.sqrt(3.0 / fan)
        torch.nn.init.uniform_(self.latent_routing, -bound, bound)
        #mask = utils.load_synthconv_mask('data/heat/3/default/mask.npy')
        #self.latent_routing.data = mask.to(dtype=torch.float32)
        #self.latent_routing.requires_grad = False
