import torch
from typing import Any, Tuple
from models.smoe_config import SpatialMoEConfig
from utils.misc import get_incorrect_indices, unscale_quantity


class SMoERouting(torch.autograd.Function):
    """Implement SMoE routing.

    This uses the raw routing weights from an SMoE gate and the output
    from experts (in its full, non-sparse form) and picks the winning
    experts, optionally scaling them by the winning routing weights.

    The routing is done based on the top k values in the routing
    weights at each point.

    This is a bit hard to implement with normal differentiable PyTorch
    ops, so here we manually implement the forward and backward parts
    of this routing.

    This is essentially a straight-through estimator. Gradients are
    sent only to experts through the positions they are selected at.
    The gate, via the routing weights, also only receives gradients
    through such points.

    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx: Any,
                experts: torch.Tensor,              # (B, T, N, E, F)
                routing_weights: torch.Tensor,      # (B, T, N, E)
                smoe_config: SpatialMoEConfig,
                save_module: Any = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        ctx.smoe_config = smoe_config
        ctx.save_module = save_module

        B, T, N, E, F = experts.shape
        K = smoe_config.out_planes

        # 取 top-K expert index 和 value：在 expert 维度
        vals, indices = routing_weights.topk(k=K, dim=3)  # [B, T, N, K]

        ctx.indices = indices
        ctx.save_for_backward(experts, routing_weights)

        # 构造 sparse 路由图 [B, T, N, E] => [B, T, N, E]
        routing_map = torch.zeros_like(routing_weights).scatter(dim=3, index=indices, src=vals)
        ctx.mark_non_differentiable(routing_map)
        ctx.mark_non_differentiable(indices)

        if smoe_config.unweighted:
            # 使用 index 选中 expert（必须扩展 indices 来匹配 shape）
            idx_expanded = indices.unsqueeze(-1).expand(B, T, N, K, F)  # [B, T, N, K, F]
            selected = torch.gather(experts, dim=3, index=idx_expanded)  # [B, T, N, K, F]
        else:
            routing_map_expanded = routing_map.unsqueeze(-1)  # [B, T, N, E, 1]
            scaled_experts = experts * routing_map_expanded  # [B, T, N, E, F]

            idx_expanded = indices.unsqueeze(-1).expand(B, T, N, K, F)  # [B, T, N, K, F]
            selected = torch.gather(scaled_experts, dim=3, index=idx_expanded)  # [B, T, N, K, F]

        return selected, routing_map, indices

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx: Any,
                grad_selected: torch.Tensor,  # [B, T, N, K, F]
                grad_routing_map: torch.Tensor,
                grad_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None, None]:

        experts, routing_weights = ctx.saved_tensors
        B, T, N, E, F = experts.shape
        K = ctx.indices.shape[-1]

        smoe_config = ctx.smoe_config

        scattered_grads = torch.zeros_like(experts, dtype=grad_selected.dtype).scatter(
            dim=3, index=ctx.indices.unsqueeze(-1).expand(B, T, N, K, F), src=grad_selected)

        grad_experts = grad_routing_weights = None

        if ctx.save_module and smoe_config.save_error_signal:
            if not smoe_config.unweighted:
                selected_experts = torch.gather(experts, dim=3, index=ctx.indices.unsqueeze(-1).expand_as(grad_selected))
                ctx.save_module.saved_error_signal = grad_selected * selected_experts
            else:
                ctx.save_module.saved_error_signal = grad_selected

        if ctx.needs_input_grad[0]:
            grad_experts = scattered_grads if smoe_config.unweighted else scattered_grads * routing_weights.unsqueeze(-1)

        if ctx.needs_input_grad[1] and not smoe_config.block_gate_grad:
            grad_routing_weights = (
                scattered_grads.clone().sum(dim=-1)
                if smoe_config.unweighted else
                (scattered_grads * experts).sum(dim=-1)
            )

        # 可选：dampen_expert_error 等略去
        return grad_experts, grad_routing_weights, None, None