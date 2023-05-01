from typing import Optional, List, Union, Sequence, Callable

import gym
import torch
import torch.nn as nn
from gym.spaces import Dict as SpaceDict

from allenact.algorithms.onpolicy_sync.policy import ActorCriticModel, ObservationType, DistributionType
from allenact.base_abstractions.distributions import Distr
from allenact.base_abstractions.misc import Memory
from projects.scmb.models.basic_models import RNNStateEncoder, LinearActorCritic
from projects.scmb.models.visual_nav_models import (
    VisualNavActorCritic,
    SCMBVisualNavActorCritic,
    FusionType,
)
from allenact.utils.model_utils import FeatureEmbedding


class ResNetPointNavActorCritic(VisualNavActorCritic):
    """Use raw image as observation to the agent."""

    def __init__(
            # base params
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            state_goal_sensor_uuid: str,
            hidden_size=512,
            num_rnn_layers=1,
            rnn_type="GRU",
            add_prev_actions=False,
            add_prev_action_null_token=False,
            action_embed_size=4,
            multiple_beliefs=False,
            beliefs_fusion: Optional[FusionType] = None,
            auxiliary_uuids: Optional[Sequence[str]] = None,
            # custom params
            rgb_resnet_preprocessor_uuid: Optional[str] = None,
            depth_resnet_preprocessor_uuid: Optional[str] = None,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            # perception backbone params,
            backbone="gnresnet18",
    ):
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            hidden_size=hidden_size,
            multiple_beliefs=multiple_beliefs,
            beliefs_fusion=beliefs_fusion,
            auxiliary_uuids=auxiliary_uuids,
        )

        self.state_goal_sensor_uuid = state_goal_sensor_uuid
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coordinate_embedding_size = coordinate_embedding_dim
        else:
            self.coordinate_embedding_size = coordinate_dims

        self.sensor_fusion = False
        self.rgb_resnet_preprocessor_uuid = rgb_resnet_preprocessor_uuid
        self.depth_resnet_preprocessor_uuid = depth_resnet_preprocessor_uuid
        if rgb_resnet_preprocessor_uuid is not None and depth_resnet_preprocessor_uuid is not None:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True

        self.backbone = backbone
        self.visual_embedding = nn.Linear(self.observation_space["rgb_resnet"].shape[0], self.recurrent_hidden_state_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.create_state_encoders(
            obs_embed_size=self.goal_visual_encoder_output_dims,
            num_rnn_layers=num_rnn_layers,
            rnn_type=rnn_type,
            add_prev_actions=add_prev_actions,
            add_prev_action_null_token=add_prev_action_null_token,
            prev_action_embed_size=action_embed_size,
        )

        self.create_actorcritic_head()

        self.create_aux_models(
            obs_embed_size=self.goal_visual_encoder_output_dims,
            action_embed_size=action_embed_size,
        )

        self.train()

    @property
    def is_blind(self):
        return False

    @property
    def goal_visual_encoder_output_dims(self):
        dims = self.coordinate_embedding_size
        if self.is_blind:
            return dims
        return dims + self.recurrent_hidden_state_size

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.state_goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.state_goal_sensor_uuid].to(torch.float32)

    def forward_encoder(
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
    ) -> torch.FloatTensor:
        target_encoding = self.get_target_coordinates_encoding(observations)
        obs_embeds: Union[torch.Tensor, List[torch.Tensor]]
        obs_embeds = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_embedding(observations[self.rgb_resnet_preprocessor_uuid])
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            obs_embeds = [perception_embed] + obs_embeds

        obs_embeds = torch.cat(obs_embeds, dim=-1)
        return obs_embeds


class SCMBResNetPointNavActorCritic(SCMBVisualNavActorCritic):
    """Use raw image as observation to the agent."""

    def __init__(
            # base params
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            state_goal_sensor_uuid: str,
            prev_state_goal_sensor_uuid: str,
            hidden_size=512,
            num_rnn_layers=1,
            rnn_type="GRU",
            add_prev_actions=False,
            add_prev_action_null_token=False,
            action_embed_size=4,
            multiple_beliefs=False,
            beliefs_fusion: Optional[FusionType] = None,
            auxiliary_uuids: Optional[Sequence[str]] = None,
            # custom params
            rgb_resnet_preprocessor_uuid: Optional[str] = None,
            depth_resnet_preprocessor_uuid: Optional[str] = None,
            prev_rgb_resnet_preprocessor_uuid: Optional[str] = None,
            prev_depth_resnet_preprocessor_uuid: Optional[str] = None,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            embed_state_changes=False,
            state_change_embedding_dim=16,
            state_change_dims=4,
            state_changes_pred_dims=4,
            certain_action_state_change_embed=[],
            use_transformer_head=False,
            num_decoders=6,
            head_type: Callable[..., ActorCriticModel[Distr]] = LinearActorCritic,
            is_vo_baseline: bool = False,
            # perception backbone params,
            backbone="gnresnet18",
    ):
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            hidden_size=hidden_size,
            multiple_beliefs=multiple_beliefs,
            beliefs_fusion=beliefs_fusion,
            auxiliary_uuids=auxiliary_uuids,
            use_transformer_head=use_transformer_head,
            num_decoders=num_decoders,
        )

        self.head_type = head_type
        self.is_vo_baseline = is_vo_baseline
        self.state_goal_sensor_uuid = state_goal_sensor_uuid
        self.prev_state_goal_sensor_uuid = prev_state_goal_sensor_uuid
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coordinate_embedding_size = coordinate_embedding_dim
        else:
            self.coordinate_embedding_size = coordinate_dims

        self.sensor_fusion = False
        self.rgb_resnet_preprocessor_uuid = rgb_resnet_preprocessor_uuid
        self.depth_resnet_preprocessor_uuid = depth_resnet_preprocessor_uuid
        if rgb_resnet_preprocessor_uuid is not None and depth_resnet_preprocessor_uuid is not None:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True

        self.backbone = backbone
        self.visual_embedding = nn.Linear(self.observation_space["rgb_resnet"].shape[0], self.recurrent_hidden_state_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.create_state_encoders(
            obs_embed_size=self.goal_visual_encoder_output_dims,
            num_rnn_layers=num_rnn_layers,
            rnn_type=rnn_type,
            add_prev_actions=add_prev_actions,
            add_prev_action_null_token=add_prev_action_null_token,
            prev_action_embed_size=action_embed_size,
        )

        self.create_actorcritic_head()

        self.create_aux_models(
            obs_embed_size=self.goal_visual_encoder_output_dims,
            action_embed_size=action_embed_size,
        )

        # Hao: Modules for state changes
        self.state_changes_pred_dims = state_changes_pred_dims
        self.embed_state_changes = embed_state_changes
        if self.embed_state_changes:
            self.state_change_embedding_size = state_change_embedding_dim
        else:
            self.state_change_embedding_size = state_change_dims
        self.state_changes_sensor_funsion = False
        self.prev_rgb_resnet_preprocessor_uuid = prev_rgb_resnet_preprocessor_uuid
        self.prev_depth_resnet_preprocessor_uuid = prev_depth_resnet_preprocessor_uuid
        if prev_rgb_resnet_preprocessor_uuid is not None or prev_depth_resnet_preprocessor_uuid is not None:
            self.state_changes_sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.state_changes_sensor_funsion = True

        if self.embed_state_changes:
            self.state_changes_embedding = nn.Sequential(
                nn.Linear(state_change_dims, state_change_embedding_dim),
                nn.ReLU(),
                nn.Linear(state_change_embedding_dim, state_change_embedding_dim),
                nn.ReLU(),
                nn.Linear(state_change_embedding_dim, state_change_embedding_dim),
                nn.ReLU(),
            )

        self.state_change_encoder = RNNStateEncoder(
            input_size=self.state_changes_encoder_output_dims,
            hidden_size=self._hidden_size // 4,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            trainable_masked_hidden_state=True,
        )
        self.certain_action_state_change_embed = certain_action_state_change_embed
        if len(self.certain_action_state_change_embed) > 0:
            self.certain_action_embedder = FeatureEmbedding(
                input_size=self.action_space.n,
                output_size=self._hidden_size // 4,
            )

        self.fc_pred = nn.Linear(self._hidden_size // 4, self.state_changes_pred_dims)
        if is_vo_baseline:
            self.fc_rep_embed = nn.Linear(self.state_changes_pred_dims, self._hidden_size // 4)
        else:
            self.fc_rep_embed = nn.Linear(self._hidden_size // 4, self._hidden_size // 4)
        # self.fc_rep_embed_expand = nn.Linear(self._hidden_size // 4, self._hidden_size)
        self.fc_pred_embed = nn.Linear(action_space.n * self._hidden_size // 4, self._hidden_size // 4)
        self.relu = nn.ReLU()

        self.train()

    @property
    def is_blind(self):
        return False

    @property
    def goal_visual_encoder_output_dims(self):
        dims = self.coordinate_embedding_size
        if self.is_blind:
            return dims
        return dims + self.recurrent_hidden_state_size + self.recurrent_hidden_state_size // 4

    @property
    def state_changes_encoder_output_dims(self):
        dims = self.state_change_embedding_size
        if self.is_blind:
            return dims
        return dims + self.recurrent_hidden_state_size

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.state_goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.state_goal_sensor_uuid].to(torch.float32)

    def get_state_change_encoding(self, observations):
        state = observations[self.state_goal_sensor_uuid].to(torch.float32)
        prev_state = observations[self.prev_state_goal_sensor_uuid].to(torch.float32)
        two_states = torch.cat([state, prev_state], dim=-1)
        dummy_two_states = torch.cat([state, state], dim=-1)
        if self.embed_state_changes:
            return self.state_changes_embedding(two_states), self.state_changes_embedding(dummy_two_states)
        else:
            return two_states, dummy_two_states

    def forward_encoder(
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
    ) -> torch.FloatTensor:
        target_encoding = self.get_target_coordinates_encoding(observations)
        state_change_embeds, dummy_state_change_embeds = self.get_state_change_encoding(observations)
        obs_embeds: Union[torch.Tensor, List[torch.Tensor]]
        obs_embeds = [target_encoding]
        state_change_embeds = [state_change_embeds]
        dummy_state_change_embeds = [dummy_state_change_embeds]

        if not self.is_blind:
            perception_embed = self.visual_embedding(observations[self.rgb_resnet_preprocessor_uuid])
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            obs_embeds = [perception_embed] + obs_embeds

            prev_perception_embed = self.visual_embedding(observations[self.prev_rgb_resnet_preprocessor_uuid])
            if self.sensor_fusion:
                prev_perception_embed = self.sensor_fuser(prev_perception_embed)

            state_change_perception_embed = torch.cat([
                perception_embed, prev_perception_embed
            ], dim=-1)
            state_change_perception_embed = self.state_changes_sensor_fuser(state_change_perception_embed)

            dummy_state_change_perception_embed = torch.cat([
                perception_embed, perception_embed
            ], dim=-1)
            dummy_state_change_perception_embed = self.state_changes_sensor_fuser(dummy_state_change_perception_embed)

            state_change_embeds = [state_change_perception_embed] + state_change_embeds
            dummy_state_change_embeds = [dummy_state_change_perception_embed] + dummy_state_change_embeds

        state_change_embeds = torch.cat(state_change_embeds, dim=-1)
        dummy_state_change_embeds = torch.cat(dummy_state_change_embeds, dim=-1)

        nsteps, nsamplers = masks.shape[:2]
        next_state_pred = []
        sc_memory_return = []
        sc_rep = []
        for na in range(self.num_actions):
            action_mask = (prev_actions == na).float().unsqueeze(-1)
            if len(self.certain_action_state_change_embed) < self.num_actions:
                sc_input = state_change_embeds * action_mask + dummy_state_change_embeds * (1 - action_mask)
                state_change_rnn_out, state_change_mem_return = self.state_change_encoder(
                    x=sc_input,
                    hidden_states=memory.tensor(self.sc_memory_key[na]),
                    masks=masks,
                )
                sc_memory_return.append(state_change_mem_return)

            if len(self.certain_action_state_change_embed) > 0:
                if na in self.certain_action_state_change_embed:
                    special_action_idx = (torch.ones_like(action_mask) * na).long().squeeze(-1)
                    state_change_rnn_out = self.certain_action_embedder(special_action_idx)

            pred = self.fc_pred(state_change_rnn_out)
            next_state_pred.append(pred)
            if self.is_vo_baseline:
                sc_rep.append(self.fc_rep_embed(pred).unsqueeze(-1))
            else:
                sc_rep.append(self.fc_rep_embed(state_change_rnn_out).unsqueeze(-1))
        next_state_pred = torch.cat(next_state_pred, dim=-1)
        next_state_rep = torch.cat(sc_rep, dim=-1).transpose(2, 3)
        next_state_embeds = self.relu(self.fc_pred_embed(
            self.relu(next_state_rep).transpose(2, 3).contiguous().view(nsteps, nsamplers, -1)
        ))
        # next_state_rep = self.relu(self.fc_rep_embed_expand(next_state_rep))
        obs_embeds += [next_state_embeds]

        obs_embeds = torch.cat(obs_embeds, dim=-1)

        if len(self.certain_action_state_change_embed) < self.num_actions:
            for na in range(self.num_actions):
                memory.set_tensor(self.sc_memory_key[na], sc_memory_return[na])

        return obs_embeds, next_state_rep, next_state_pred, memory
