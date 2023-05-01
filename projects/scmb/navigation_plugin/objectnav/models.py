"""Baseline models for use in the object navigation task.

Object navigation is currently available as a Task in AI2-THOR and
Facebook's Habitat.
"""
from typing import Optional, List, Dict, cast, Tuple, Sequence, Callable
from allenact.utils.system import get_logger

import gym
import torch
import torch.nn as nn
from gym.spaces import Dict as SpaceDict
import numpy as np

from allenact.algorithms.onpolicy_sync.policy import ActorCriticModel, ObservationType, DistributionType
from allenact.base_abstractions.distributions import Distr
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.embodiedai.models import resnet as resnet
from allenact.embodiedai.models.basic_models import SimpleCNN
from allenact.embodiedai.models.visual_nav_models import (
    VisualNavActorCritic,
    FusionType,
)
from allenact.embodiedai.aux_losses.losses import MultiAuxTaskNegEntropyLoss
from projects.scmb.models.basic_models import RNNStateEncoder, LinearActorCritic
from projects.scmb.models.transformer_lm import TransformerHeadEN
from allenact.utils.model_utils import FeatureEmbedding


class CatObservations(nn.Module):
    def __init__(self, ordered_uuids: Sequence[str], dim: int):
        super().__init__()
        assert len(ordered_uuids) != 0

        self.ordered_uuids = ordered_uuids
        self.dim = dim

    def forward(self, observations: ObservationType):
        if len(self.ordered_uuids) == 1:
            return observations[self.ordered_uuids[0]]
        return torch.cat(
            [observations[uuid] for uuid in self.ordered_uuids], dim=self.dim
        )


class ObjectNavActorCritic(VisualNavActorCritic):
    """Baseline recurrent actor critic model for object-navigation.

    # Attributes
    action_space : The space of actions available to the agent. Currently only discrete
        actions are allowed (so this space will always be of type `gym.spaces.Discrete`).
    observation_space : The observation space expected by the agent. This observation space
        should include (optionally) 'rgb' images and 'depth' images and is required to
        have a component corresponding to the goal `goal_sensor_uuid`.
    goal_sensor_uuid : The uuid of the sensor of the goal object. See `GoalObjectTypeThorSensor`
        as an example of such a sensor.
    hidden_size : The hidden size of the GRU RNN.
    object_type_embedding_dim: The dimensionality of the embedding corresponding to the goal
        object type.
    """

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        # RNN
        hidden_size=512,
        num_rnn_layers=1,
        rnn_type="GRU",
        add_prev_actions=False,
        add_prev_action_null_token=False,
        action_embed_size=6,
        # Aux loss
        multiple_beliefs=False,
        beliefs_fusion: Optional[FusionType] = None,
        auxiliary_uuids: Optional[Sequence[str]] = None,
        # below are custom params
        rgb_uuid: Optional[str] = None,
        depth_uuid: Optional[str] = None,
        object_type_embedding_dim=8,
        trainable_masked_hidden_state: bool = False,
        # perception backbone params,
        backbone="gnresnet18",
        resnet_baseplanes=32,
    ):
        """Initializer.

        See class documentation for parameter definitions.
        """
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            hidden_size=hidden_size,
            multiple_beliefs=multiple_beliefs,
            beliefs_fusion=beliefs_fusion,
            auxiliary_uuids=auxiliary_uuids,
        )

        self.rgb_uuid = rgb_uuid
        self.depth_uuid = depth_uuid

        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_object_types = self.observation_space.spaces[self.goal_sensor_uuid].n
        self.object_type_embedding_size = object_type_embedding_dim

        self.backbone = backbone
        if backbone == "simple_cnn":
            self.visual_encoder = SimpleCNN(
                observation_space=observation_space,
                output_size=hidden_size,
                rgb_uuid=rgb_uuid,
                depth_uuid=depth_uuid,
            )
            self.visual_encoder_output_size = hidden_size
            assert self.is_blind == self.visual_encoder.is_blind
        elif backbone == "gnresnet18":  # resnet family
            self.visual_encoder = resnet.GroupNormResNetEncoder(
                observation_space=observation_space,
                output_size=hidden_size,
                rgb_uuid=rgb_uuid,
                depth_uuid=depth_uuid,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
            )
            self.visual_encoder_output_size = hidden_size
            assert self.is_blind == self.visual_encoder.is_blind
        elif backbone in ["identity", "projection"]:
            good_uuids = [
                uuid for uuid in [self.rgb_uuid, self.depth_uuid] if uuid is not None
            ]
            cat_model = CatObservations(ordered_uuids=good_uuids, dim=-1,)
            after_cat_size = sum(
                observation_space[uuid].shape[-1] for uuid in good_uuids
            )
            if backbone == "identity":
                self.visual_encoder = cat_model
                self.visual_encoder_output_size = after_cat_size
            else:
                self.visual_encoder = nn.Sequential(
                    cat_model, nn.Linear(after_cat_size, hidden_size), nn.ReLU(True)
                )
                self.visual_encoder_output_size = hidden_size

        else:
            raise NotImplementedError

        self.create_state_encoders(
            obs_embed_size=self.goal_visual_encoder_output_dims,
            num_rnn_layers=num_rnn_layers,
            rnn_type=rnn_type,
            add_prev_actions=add_prev_actions,
            add_prev_action_null_token=add_prev_action_null_token,
            prev_action_embed_size=action_embed_size,
            trainable_masked_hidden_state=trainable_masked_hidden_state,
        )

        self.create_actorcritic_head()

        self.create_aux_models(
            obs_embed_size=self.goal_visual_encoder_output_dims,
            action_embed_size=action_embed_size,
        )

        self.object_type_embedding = nn.Embedding(
            num_embeddings=self._n_object_types,
            embedding_dim=object_type_embedding_dim,
        )

        self.train()

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.rgb_uuid is None and self.depth_uuid is None

    @property
    def goal_visual_encoder_output_dims(self):
        dims = self.object_type_embedding_size
        if self.is_blind:
            return dims
        return dims + self.visual_encoder_output_size

    def get_object_type_encoding(
        self, observations: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Get the object type encoding from input batched observations."""
        # noinspection PyTypeChecker
        return self.object_type_embedding(  # type:ignore
            observations[self.goal_sensor_uuid].to(torch.int64)
        )

    def forward_encoder(self, observations: ObservationType) -> torch.Tensor:
        target_encoding = self.get_object_type_encoding(
            cast(Dict[str, torch.Tensor], observations)
        )
        obs_embeds = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            obs_embeds = [perception_embed] + obs_embeds

        obs_embeds = torch.cat(obs_embeds, dim=-1)
        return obs_embeds


class ResnetTensorNavActorCritic(VisualNavActorCritic):
    def __init__(
            # base params
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            hidden_size=512,
            num_rnn_layers=1,
            rnn_type="GRU",
            add_prev_actions=False,
            add_prev_action_null_token=False,
            action_embed_size=6,
            multiple_beliefs=False,
            beliefs_fusion: Optional[FusionType] = None,
            auxiliary_uuids: Optional[List[str]] = None,
            # custom params
            rgb_resnet_preprocessor_uuid: Optional[str] = None,
            depth_resnet_preprocessor_uuid: Optional[str] = None,
            inter0_rgb_resnet_preprocessor_uuid: Optional[str] = None,
            goal_dims: int = 32,
            resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
            combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ):
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            hidden_size=hidden_size,
            multiple_beliefs=multiple_beliefs,
            beliefs_fusion=beliefs_fusion,
            auxiliary_uuids=auxiliary_uuids,
        )

        self.cnt = 0
        if (
                rgb_resnet_preprocessor_uuid is None
                or depth_resnet_preprocessor_uuid is None
        ):
            resnet_preprocessor_uuid = (
                rgb_resnet_preprocessor_uuid
                if rgb_resnet_preprocessor_uuid is not None
                else depth_resnet_preprocessor_uuid
            )
            self.goal_visual_encoder = ResnetTensorGoalEncoder(
                self.observation_space,
                goal_sensor_uuid,
                resnet_preprocessor_uuid,
                goal_dims,
                resnet_compressor_hidden_out_dims,
                combiner_hidden_out_dims,
            )
            self.cnt += 1
            inter0_resnet_preprocessor_uuid = (
                inter0_rgb_resnet_preprocessor_uuid
                if inter0_rgb_resnet_preprocessor_uuid is not None
                else None
            )
            if inter0_resnet_preprocessor_uuid is not None:
                self.inter0_goal_visual_encoder = ResnetTensorGoalEncoder(
                    self.observation_space,
                    goal_sensor_uuid,
                    inter0_resnet_preprocessor_uuid,
                    goal_dims,
                    resnet_compressor_hidden_out_dims,
                    combiner_hidden_out_dims,
                )
                self.cnt += 1
        else:
            self.goal_visual_encoder = ResnetDualTensorGoalEncoder(  # type:ignore
                self.observation_space,
                goal_sensor_uuid,
                rgb_resnet_preprocessor_uuid,
                depth_resnet_preprocessor_uuid,
                goal_dims,
                resnet_compressor_hidden_out_dims,
                combiner_hidden_out_dims,
            )

        if self.cnt > 1:
            self.state_changes_sensor_fuser = nn.Linear(
                self.goal_visual_encoder.output_dims * self.cnt,
                self.goal_visual_encoder.output_dims
            )

        self.create_state_encoders(
            obs_embed_size=self.goal_visual_encoder.output_dims,
            num_rnn_layers=num_rnn_layers,
            rnn_type=rnn_type,
            add_prev_actions=add_prev_actions,
            add_prev_action_null_token=add_prev_action_null_token,
            prev_action_embed_size=action_embed_size,
        )

        self.create_actorcritic_head()

        self.create_aux_models(
            obs_embed_size=self.goal_visual_encoder.output_dims,
            action_embed_size=action_embed_size,
        )

        self.train()

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.goal_visual_encoder.is_blind

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        out = self.goal_visual_encoder(observations)
        if self.cnt > 1:
            out = torch.cat([out, self.inter0_goal_visual_encoder(observations)], dim=-1)
            out = self.state_changes_sensor_fuser(out)
        return out


class SCMBResnetTensorNavActorCritic(VisualNavActorCritic):
    def __init__(
            # base params
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            hidden_size=512,
            num_rnn_layers=1,
            rnn_type="GRU",
            add_prev_actions=False,
            add_prev_action_null_token=False,
            action_embed_size=6,
            multiple_beliefs=False,
            beliefs_fusion: Optional[FusionType] = None,
            auxiliary_uuids: Optional[List[str]] = None,
            # custom params
            rgb_resnet_preprocessor_uuid: Optional[str] = None,
            depth_resnet_preprocessor_uuid: Optional[str] = None,
            prev_rgb_resnet_preprocessor_uuid: Optional[str] = None,
            prev_depth_resnet_preprocessor_uuid: Optional[str] = None,
            inter0_rgb_resnet_preprocessor_uuid: Optional[str] = None,
            inter0_depth_resnet_preprocessor_uuid: Optional[str] = None,
            inter1_rgb_resnet_preprocessor_uuid: Optional[str] = None,
            inter1_depth_resnet_preprocessor_uuid: Optional[str] = None,
            inter2_rgb_resnet_preprocessor_uuid: Optional[str] = None,
            inter2_depth_resnet_preprocessor_uuid: Optional[str] = None,
            inter3_rgb_resnet_preprocessor_uuid: Optional[str] = None,
            inter3_depth_resnet_preprocessor_uuid: Optional[str] = None,
            goal_dims: int = 32,
            resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
            combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
            # scmb
            state_changes_pred_dims=4,
            certain_action_state_change_embed=[],
            use_transformer_head: bool = False,
            num_decoders: int = 6,
            head_type: Callable[..., ActorCriticModel[Distr]] = LinearActorCritic,
            is_vo_baseline: bool = False,
    ):
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            hidden_size=hidden_size,
            multiple_beliefs=multiple_beliefs,
            beliefs_fusion=beliefs_fusion,
            auxiliary_uuids=auxiliary_uuids,
        )

        self.is_vo_baseline = is_vo_baseline
        self.cnt = 0
        if (
                rgb_resnet_preprocessor_uuid is None
                or depth_resnet_preprocessor_uuid is None
        ):
            resnet_preprocessor_uuid = (
                rgb_resnet_preprocessor_uuid
                if rgb_resnet_preprocessor_uuid is not None
                else depth_resnet_preprocessor_uuid
            )
            self.goal_visual_encoder = ResnetTensorGoalEncoder(
                self.observation_space,
                goal_sensor_uuid,
                resnet_preprocessor_uuid,
                goal_dims,
                resnet_compressor_hidden_out_dims,
                combiner_hidden_out_dims,
            )
            self.cnt += 1
            prev_resnet_preprocessor_uuid = (
                prev_rgb_resnet_preprocessor_uuid
                if prev_rgb_resnet_preprocessor_uuid is not None
                else prev_depth_resnet_preprocessor_uuid
            )
            self.prev_goal_visual_encoder = ResnetTensorGoalEncoder(
                self.observation_space,
                goal_sensor_uuid,
                prev_resnet_preprocessor_uuid,
                goal_dims,
                resnet_compressor_hidden_out_dims,
                combiner_hidden_out_dims,
            )
            self.cnt += 1
            inter0_resnet_preprocessor_uuid = (
                inter0_rgb_resnet_preprocessor_uuid
                if inter0_rgb_resnet_preprocessor_uuid is not None
                else inter0_depth_resnet_preprocessor_uuid
            )
            if inter0_resnet_preprocessor_uuid is not None:
                self.inter0_goal_visual_encoder = ResnetTensorGoalEncoder(
                    self.observation_space,
                    goal_sensor_uuid,
                    inter0_resnet_preprocessor_uuid,
                    goal_dims,
                    resnet_compressor_hidden_out_dims,
                    combiner_hidden_out_dims,
                )
                self.cnt += 1
            inter1_resnet_preprocessor_uuid = (
                inter1_rgb_resnet_preprocessor_uuid
                if inter1_rgb_resnet_preprocessor_uuid is not None
                else inter1_depth_resnet_preprocessor_uuid
            )
            if inter1_resnet_preprocessor_uuid is not None:
                self.inter1_goal_visual_encoder = ResnetTensorGoalEncoder(
                    self.observation_space,
                    goal_sensor_uuid,
                    inter1_resnet_preprocessor_uuid,
                    goal_dims,
                    resnet_compressor_hidden_out_dims,
                    combiner_hidden_out_dims,
                )
                self.cnt += 1
            inter2_resnet_preprocessor_uuid = (
                inter2_rgb_resnet_preprocessor_uuid
                if inter2_rgb_resnet_preprocessor_uuid is not None
                else inter2_depth_resnet_preprocessor_uuid
            )
            if inter2_resnet_preprocessor_uuid is not None:
                self.inter2_goal_visual_encoder = ResnetTensorGoalEncoder(
                    self.observation_space,
                    goal_sensor_uuid,
                    inter2_resnet_preprocessor_uuid,
                    goal_dims,
                    resnet_compressor_hidden_out_dims,
                    combiner_hidden_out_dims,
                )
                self.cnt += 1
            inter3_resnet_preprocessor_uuid = (
                inter3_rgb_resnet_preprocessor_uuid
                if inter3_rgb_resnet_preprocessor_uuid is not None
                else inter3_depth_resnet_preprocessor_uuid
            )
            if inter3_resnet_preprocessor_uuid is not None:
                self.inter3_goal_visual_encoder = ResnetTensorGoalEncoder(
                    self.observation_space,
                    goal_sensor_uuid,
                    inter3_resnet_preprocessor_uuid,
                    goal_dims,
                    resnet_compressor_hidden_out_dims,
                    combiner_hidden_out_dims,
                )
                self.cnt += 1
        else:
            raise NotImplementedError

        self.create_state_encoders(
            obs_embed_size=self.goal_visual_encoder.output_dims + self.recurrent_hidden_state_size // 4,
            num_rnn_layers=num_rnn_layers,
            rnn_type=rnn_type,
            add_prev_actions=add_prev_actions,
            add_prev_action_null_token=add_prev_action_null_token,
            prev_action_embed_size=action_embed_size,
        )

        self.create_aux_models(
            obs_embed_size=self.goal_visual_encoder.output_dims,
            action_embed_size=action_embed_size,
        )

        self.num_actions = action_space.n
        self.sc_memory_key = ["sc_{}".format(na) for na in range(self.num_actions)]
        self.use_transformer_head = use_transformer_head
        self.transformer_head = None
        self.num_decoders = num_decoders
        self.head_type = head_type
        self.head_uuid = "{}_{}".format("rnn", "enc")

        self.create_actorcritic_head()

        # Hao: Modules for state changes
        self.state_changes_pred_dims = state_changes_pred_dims
        self.state_changes_sensor_funsion = False
        self.prev_rgb_resnet_preprocessor_uuid = prev_rgb_resnet_preprocessor_uuid
        self.prev_depth_resnet_preprocessor_uuid = prev_depth_resnet_preprocessor_uuid
        self.state_changes_sensor_fuser = nn.Linear(
            self.goal_visual_encoder.output_dims * self.cnt,
            self.goal_visual_encoder.output_dims
        )
        self.state_changes_sensor_funsion = True

        self.state_change_encoder = RNNStateEncoder(
            input_size=self.goal_visual_encoder.output_dims,
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
        self.fc_pred_embed = nn.Linear(action_space.n * self._hidden_size // 4, self._hidden_size // 4)
        self.relu = nn.ReLU()

        self.train()

    def create_actorcritic_head(self):
        if self.head_type is None:
            self.head = LinearActorCritic(
                input_uuid=self.head_uuid,
                action_space=self.action_space,
                observation_space=SpaceDict(
                    {
                        self.head_uuid: gym.spaces.Box(
                            low=np.float32(0.0), high=np.float32(1.0), shape=(self._hidden_size,)
                        )
                    }
                ),
            )
        else:
            self.head = self.head_type(
                input_uuid=self.head_uuid,
                action_space=self.action_space,
                observation_space=SpaceDict(
                    {
                        self.head_uuid: gym.spaces.Box(
                            low=np.float32(0.0), high=np.float32(1.0), shape=(self._hidden_size,)
                        )
                    }
                ),
            )
            if self.use_transformer_head:
                self.transformer_head = TransformerHeadEN(
                    nfeatures=self._hidden_size,
                    ninp=self._hidden_size // 4,
                    nhead=self._hidden_size // 32,
                    nhid=self._hidden_size,
                    dropout=0,
                    initrange=0.1,
                    ndecoder=self.num_decoders,
                )
                self.embed_belief = nn.Linear(self._hidden_size, self._hidden_size // 4)

    def _recurrent_memory_specification(self):
        main_spec = {
            memory_key: (
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
            for memory_key in self.belief_names
        }
        sc_spec = {
            self.sc_memory_key[na]: (
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size // 4),
                ),
                torch.float32,
            ) for na in range(self.num_actions)
        }
        main_spec.update(sc_spec)
        return main_spec

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.goal_visual_encoder.is_blind

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        return self.goal_visual_encoder(observations)

    def forward_prev_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        return self.prev_goal_visual_encoder(observations)

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        """Processes input batched observations to produce new actor and critic
        values. Processes input batched observations (along with prior hidden
        states, previous actions, and masks denoting which recurrent hidden
        states should be masked) and returns an `ActorCriticOutput` object
        containing the model's policy (distribution over actions) and
        evaluation of the current state (value).

        # Parameters
        observations : Batched input observations.
        memory : `Memory` containing the hidden states from initial timepoints.
        prev_actions : Tensor of previous actions taken.
        masks : Masks applied to hidden states. See `RNNStateEncoder`.
        # Returns
        Tuple of the `ActorCriticOutput` and recurrent hidden state.
        """

        # 1.1 use perception model (i.e. encoder) to get observation embeddings
        obs_embeds = self.forward_encoder(observations)
        prev_obs_embeds = self.forward_prev_encoder(observations)

        state_change_embeds = torch.cat([obs_embeds, prev_obs_embeds], dim=-1)
        dummy_state_change_embeds = torch.cat([obs_embeds, obs_embeds], dim=-1)
        if self.cnt > 2:
            inter_obs_embeds = self.inter0_goal_visual_encoder(observations)
            state_change_embeds = torch.cat([state_change_embeds, inter_obs_embeds], dim=-1)
            dummy_state_change_embeds = torch.cat([dummy_state_change_embeds, obs_embeds], dim=-1)

        if self.cnt > 3:
            inter_obs_embeds = self.inter1_goal_visual_encoder(observations)
            state_change_embeds = torch.cat([state_change_embeds, inter_obs_embeds], dim=-1)
            dummy_state_change_embeds = torch.cat([dummy_state_change_embeds, obs_embeds], dim=-1)

        if self.cnt > 4:
            inter_obs_embeds = self.inter2_goal_visual_encoder(observations)
            state_change_embeds = torch.cat([state_change_embeds, inter_obs_embeds], dim=-1)
            dummy_state_change_embeds = torch.cat([dummy_state_change_embeds, obs_embeds], dim=-1)

        if self.cnt > 5:
            inter_obs_embeds = self.inter3_goal_visual_encoder(observations)
            state_change_embeds = torch.cat([state_change_embeds, inter_obs_embeds], dim=-1)
            dummy_state_change_embeds = torch.cat([dummy_state_change_embeds, obs_embeds], dim=-1)

        state_change_embeds = self.state_changes_sensor_fuser(state_change_embeds)
        dummy_state_change_embeds = self.state_changes_sensor_fuser(dummy_state_change_embeds)

        nsteps, nsamplers = masks.shape[:2]
        next_state_pred = []
        sc_memory_return = []
        sc_rep = []
        for na in range(self.num_actions):
            action_mask = (prev_actions == na).float().unsqueeze(-1)
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

        for na in range(self.num_actions):
            memory.set_tensor(self.sc_memory_key[na], sc_memory_return[na])

        obs_embeds = torch.cat([obs_embeds, next_state_embeds], dim=-1)

        # 1.2 use embedding model to get prev_action embeddings
        if self.prev_action_embedder.input_size == self.action_space.n + 1:
            # In this case we have a unique embedding for the start of an episode
            prev_actions_embeds = self.prev_action_embedder(
                torch.where(
                    condition=0 != masks.view(*prev_actions.shape),
                    input=prev_actions + 1,
                    other=torch.zeros_like(prev_actions),
                )
            )
        else:
            prev_actions_embeds = self.prev_action_embedder(prev_actions)
        joint_embeds = torch.cat((obs_embeds, prev_actions_embeds), dim=-1)  # (T, N, *)

        # 2. use RNNs to get single/multiple beliefs
        beliefs_dict = {}
        for key, model in self.state_encoders.items():
            beliefs_dict[key], rnn_hidden_states = model(
                joint_embeds, memory.tensor(key), masks
            )
            memory.set_tensor(key, rnn_hidden_states)  # update memory here

        # 3. fuse beliefs for multiple belief models
        beliefs, task_weights = self.fuse_beliefs(
            beliefs_dict, obs_embeds
        )  # fused beliefs

        # 4. prepare output
        extras = (
            {
                aux_uuid: {
                    "beliefs": (
                        beliefs_dict[aux_uuid] if self.multiple_beliefs else beliefs
                    ),
                    "obs_embeds": obs_embeds,
                    "aux_model": (
                        self.aux_models[aux_uuid]
                        if aux_uuid in self.aux_models
                        else None
                    ),
                }
                for aux_uuid in self.auxiliary_uuids
            }
            if self.auxiliary_uuids is not None
            else {}
        )

        if self.multiple_beliefs:
            extras[MultiAuxTaskNegEntropyLoss.UUID] = task_weights

        if self.use_transformer_head:
            representation = next_state_rep
            beliefs = self.embed_belief(beliefs)
            rep = torch.cat([representation, beliefs.unsqueeze(2)], dim=2)
            n_steps, nsamplers, nactions_value, hidden_dims = rep.shape
            transformer_out = self.transformer_head(
                rep.view(n_steps * nsamplers, nactions_value, hidden_dims).transpose(0, 1)
            ).transpose(0, 1).view(n_steps, nsamplers, nactions_value, hidden_dims * 4)
            beliefs = transformer_out

        actor_critic_output, _ = self.head(
            observations={self.head_uuid: beliefs},
            memory=None,
            prev_actions=prev_actions,
            masks=masks,
        )
        actor_critic_output.extras = extras
        actor_critic_output.extras.update({"model_base_pred": next_state_pred})

        return actor_critic_output, memory


class ResnetTensorGoalEncoder(nn.Module):
    def __init__(
            self,
            observation_spaces: SpaceDict,
            goal_sensor_uuid: str,
            resnet_preprocessor_uuid: str,
            goal_embed_dims: int = 32,
            resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
            combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ) -> None:
        super().__init__()
        self.goal_uuid = goal_sensor_uuid
        self.resnet_uuid = resnet_preprocessor_uuid
        self.goal_embed_dims = goal_embed_dims
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims

        self.goal_space = observation_spaces.spaces[self.goal_uuid]
        if isinstance(self.goal_space, gym.spaces.Discrete):
            self.embed_goal = nn.Embedding(
                num_embeddings=self.goal_space.n, embedding_dim=self.goal_embed_dims,
            )
        elif isinstance(self.goal_space, gym.spaces.Box):
            self.embed_goal = nn.Linear(self.goal_space.shape[-1], self.goal_embed_dims)
        else:
            raise NotImplementedError

        self.blind = self.resnet_uuid not in observation_spaces.spaces
        if not self.blind:
            self.resnet_tensor_shape = observation_spaces.spaces[self.resnet_uuid].shape
            self.resnet_compressor = nn.Sequential(
                nn.Conv2d(self.resnet_tensor_shape[0], self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )
            self.target_obs_combiner = nn.Sequential(
                nn.Conv2d(
                    self.resnet_hid_out_dims[1] + self.goal_embed_dims,
                    self.combine_hid_out_dims[0],
                    1,
                    ),
                nn.ReLU(),
                nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
            )

    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        if self.blind:
            return self.goal_embed_dims
        else:
            return (
                    self.combine_hid_out_dims[-1]
                    * self.resnet_tensor_shape[1]
                    * self.resnet_tensor_shape[2]
            )

    def get_object_type_encoding(
            self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return cast(
            torch.FloatTensor,
            self.embed_goal(observations[self.goal_uuid].to(torch.int64)),
        )

    def compress_resnet(self, observations):
        return self.resnet_compressor(observations[self.resnet_uuid])

    def distribute_target(self, observations):
        target_emb = self.embed_goal(observations[self.goal_uuid])
        return target_emb.view(-1, self.goal_embed_dims, 1, 1).expand(
            -1, -1, self.resnet_tensor_shape[-2], self.resnet_tensor_shape[-1]
        )

    def adapt_input(self, observations):
        resnet = observations[self.resnet_uuid]
        goal = observations[self.goal_uuid]

        use_agent = False
        nagent = 1

        if len(resnet.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = resnet.shape[:3]
        else:
            nstep, nsampler = resnet.shape[:2]

        observations[self.resnet_uuid] = resnet.view(-1, *resnet.shape[-3:])
        observations[self.goal_uuid] = goal.view(-1, goal.shape[-1])

        return observations, use_agent, nstep, nsampler, nagent

    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )

        if self.blind:
            return self.embed_goal(observations[self.goal_uuid])
        embs = [
            self.compress_resnet(observations),
            self.distribute_target(observations),
        ]
        x = self.target_obs_combiner(torch.cat(embs, dim=1,))
        x = x.reshape(x.size(0), -1)  # flatten

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)


class ResnetDualTensorGoalEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        goal_sensor_uuid: str,
        rgb_resnet_preprocessor_uuid: str,
        depth_resnet_preprocessor_uuid: str,
        goal_embed_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ) -> None:
        super().__init__()
        self.goal_uuid = goal_sensor_uuid
        self.rgb_resnet_uuid = rgb_resnet_preprocessor_uuid
        self.depth_resnet_uuid = depth_resnet_preprocessor_uuid
        self.goal_embed_dims = goal_embed_dims
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims

        self.goal_space = observation_spaces.spaces[self.goal_uuid]
        if isinstance(self.goal_space, gym.spaces.Discrete):
            self.embed_goal = nn.Embedding(
                num_embeddings=self.goal_space.n, embedding_dim=self.goal_embed_dims,
            )
        elif isinstance(self.goal_space, gym.spaces.Box):
            self.embed_goal = nn.Linear(self.goal_space.shape[-1], self.goal_embed_dims)
        else:
            raise NotImplementedError

        self.blind = (
            self.rgb_resnet_uuid not in observation_spaces.spaces
            or self.depth_resnet_uuid not in observation_spaces.spaces
        )
        if not self.blind:
            self.resnet_tensor_shape = observation_spaces.spaces[
                self.rgb_resnet_uuid
            ].shape
            self.rgb_resnet_compressor = nn.Sequential(
                nn.Conv2d(self.resnet_tensor_shape[0], self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )
            self.depth_resnet_compressor = nn.Sequential(
                nn.Conv2d(self.resnet_tensor_shape[0], self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )
            self.rgb_target_obs_combiner = nn.Sequential(
                nn.Conv2d(
                    self.resnet_hid_out_dims[1] + self.goal_embed_dims,
                    self.combine_hid_out_dims[0],
                    1,
                ),
                nn.ReLU(),
                nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
            )
            self.depth_target_obs_combiner = nn.Sequential(
                nn.Conv2d(
                    self.resnet_hid_out_dims[1] + self.goal_embed_dims,
                    self.combine_hid_out_dims[0],
                    1,
                ),
                nn.ReLU(),
                nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
            )

    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        if self.blind:
            return self.goal_embed_dims
        else:
            return (
                2
                * self.combine_hid_out_dims[-1]
                * self.resnet_tensor_shape[1]
                * self.resnet_tensor_shape[2]
            )

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return cast(
            torch.FloatTensor,
            self.embed_goal(observations[self.goal_uuid].to(torch.int64)),
        )

    def compress_rgb_resnet(self, observations):
        return self.rgb_resnet_compressor(observations[self.rgb_resnet_uuid])

    def compress_depth_resnet(self, observations):
        return self.depth_resnet_compressor(observations[self.depth_resnet_uuid])

    def distribute_target(self, observations):
        target_emb = self.embed_goal(observations[self.goal_uuid])
        return target_emb.view(-1, self.goal_embed_dims, 1, 1).expand(
            -1, -1, self.resnet_tensor_shape[-2], self.resnet_tensor_shape[-1]
        )

    def adapt_input(self, observations):
        rgb = observations[self.rgb_resnet_uuid]
        depth = observations[self.depth_resnet_uuid]

        use_agent = False
        nagent = 1

        if len(rgb.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = rgb.shape[:3]
        else:
            nstep, nsampler = rgb.shape[:2]

        observations[self.rgb_resnet_uuid] = rgb.view(-1, *rgb.shape[-3:])
        observations[self.depth_resnet_uuid] = depth.view(-1, *depth.shape[-3:])
        observations[self.goal_uuid] = observations[self.goal_uuid].view(-1, 1)

        return observations, use_agent, nstep, nsampler, nagent

    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )

        if self.blind:
            return self.embed_goal(observations[self.goal_uuid])
        rgb_embs = [
            self.compress_rgb_resnet(observations),
            self.distribute_target(observations),
        ]
        rgb_x = self.rgb_target_obs_combiner(torch.cat(rgb_embs, dim=1,))
        depth_embs = [
            self.compress_depth_resnet(observations),
            self.distribute_target(observations),
        ]
        depth_x = self.depth_target_obs_combiner(torch.cat(depth_embs, dim=1,))
        x = torch.cat([rgb_x, depth_x], dim=1)
        x = x.reshape(x.shape[0], -1)  # flatten

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)
