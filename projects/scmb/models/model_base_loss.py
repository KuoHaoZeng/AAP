from typing import Dict, Union, Sequence

import torch
import typing

from allenact.algorithms.onpolicy_sync.losses.abstract_loss import AbstractActorCriticLoss
from allenact.base_abstractions.misc import ActorCriticOutput
from allenact.base_abstractions.distributions import CategoricalDistr


class Model_base_loss(AbstractActorCriticLoss):
    def __init__(self,
                 alpha: float = 3.0,
                 shift: int = 0,
                 dim: int = 6,
                 use_action: bool = False,
                 certain_action_state_change_embed: Sequence[int] = [],
                 gt_uuid: str = None):
        self.alpha = alpha
        self.shift = shift
        self.dim = dim
        self.use_action = use_action
        self.certain_action_state_change_embed = certain_action_state_change_embed
        self.gt_uuid = gt_uuid

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]],
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs
    ):
        observations = typing.cast(Dict[str, torch.Tensor], batch["observations"])
        gt_action = observations[self.gt_uuid]

        if self.use_action:
            nt = actor_critic_output.extras["model_base_pred"].shape[0]
            np = actor_critic_output.extras["model_base_pred"].shape[1]
            nd = self.dim
            pred_action = actor_critic_output.extras["model_base_pred"].view(nt, np, nd, -1)
            a_sampler = batch["actions"].long().unsqueeze(-1).unsqueeze(-1).repeat((1, 1, nd, 1))
            pred_action = torch.gather(input=pred_action, dim=-1, index=a_sampler).squeeze(-1)
        else:
            pred_action = actor_critic_output.extras["model_base_pred"]

        if gt_action.shape[0] != nt:
            gt_action = gt_action.reshape(nt, np, -1)
        gt_action = gt_action[:, :, :self.dim].to(torch.float32)
        pred_action = pred_action[:, :, :self.dim].to(torch.float32)
        masks = batch["masks"]
        if len(self.certain_action_state_change_embed) > 0:
            for na in self.certain_action_state_change_embed:
                certain_action_masks = (batch["actions"] != na).unsqueeze(-1)
                masks = masks * certain_action_masks

        if self.shift > 0:
            gt_action = gt_action[self.shift:] - gt_action[:-self.shift]
            # masks = masks[:-self.shift]
            masks = masks[self.shift:]
            pred_action = pred_action[:-self.shift]

        nt, ns = gt_action.shape[:2]
        loss = torch.nn.functional.mse_loss(pred_action.contiguous().view(nt * ns, -1),
                                            gt_action.view(nt * ns, -1),
                                            reduction="none")
        masks = masks.view(nt * ns, -1)
        loss = (loss * masks).mean(-1).sum() / masks.sum()

        return (
            loss * self.alpha,
            {"model_base_pred": loss.item(),},
        )
