from collections import OrderedDict
from typing import Tuple, Dict, Optional, List, Sequence, Union, Any
from typing import TypeVar

import gym
import torch
import torch.nn as nn
from gym.spaces.dict import Dict as SpaceDict
import numpy as np

from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    LinearCriticHead,
    LinearActorHead,
    ObservationType,
    DistributionType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.embodiedai.aux_losses.losses import MultiAuxTaskNegEntropyLoss
from projects.scmb.models.aux_models import AuxiliaryModel
from projects.scmb.models.basic_models import RNNStateEncoder, LinearActorCritic
from projects.scmb.models.transformer_lm import TransformerHead, TransformerHeadEN
from projects.scmb.models.fusion_models import Fusion
from allenact.utils.model_utils import FeatureEmbedding
from allenact.utils.system import get_logger

FusionType = TypeVar("FusionType", bound=Fusion)


class VisualNavActorCritic(ActorCriticModel[CategoricalDistr]):
    """Base class of visual navigation / manipulation (or broadly, embodied AI)
    model.

    `forward_encoder` function requires implementation.
    """

    action_space: gym.spaces.Discrete

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        hidden_size=512,
        multiple_beliefs=False,
        beliefs_fusion: Optional[FusionType] = None,
        auxiliary_uuids: Optional[List[str]] = None,
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)
        self._hidden_size = hidden_size
        assert multiple_beliefs == (beliefs_fusion is not None)
        self.multiple_beliefs = multiple_beliefs
        self.beliefs_fusion = beliefs_fusion
        self.auxiliary_uuids = auxiliary_uuids
        if isinstance(self.auxiliary_uuids, list) and len(self.auxiliary_uuids) == 0:
            self.auxiliary_uuids = None

        # Define the placeholders in init function
        self.state_encoders: Optional[nn.ModuleDict] = None
        self.aux_models: Optional[nn.ModuleDict] = None
        self.actor: Optional[LinearActorHead] = None
        self.critic: Optional[LinearCriticHead] = None
        self.prev_action_embedder: Optional[FeatureEmbedding] = None

        self.fusion_model: Optional[nn.Module] = None
        self.belief_names: Optional[Sequence[str]] = None

    def create_state_encoders(
        self,
        obs_embed_size: int,
        prev_action_embed_size: int,
        num_rnn_layers: int,
        rnn_type: str,
        add_prev_actions: bool,
        add_prev_action_null_token: bool,
        trainable_masked_hidden_state=False,
    ):
        rnn_input_size = obs_embed_size
        self.prev_action_embedder = FeatureEmbedding(
            input_size=int(add_prev_action_null_token) + self.action_space.n,
            output_size=prev_action_embed_size if add_prev_actions else 0,
        )
        if add_prev_actions:
            rnn_input_size += prev_action_embed_size

        state_encoders = OrderedDict()  # perserve insertion order in py3.6
        if self.multiple_beliefs:  # multiple belief model
            for aux_uuid in self.auxiliary_uuids:
                state_encoders[aux_uuid] = RNNStateEncoder(
                    rnn_input_size,
                    self._hidden_size,
                    num_layers=num_rnn_layers,
                    rnn_type=rnn_type,
                    trainable_masked_hidden_state=trainable_masked_hidden_state,
                )
            # create fusion model
            self.fusion_model = self.beliefs_fusion(
                hidden_size=self._hidden_size,
                obs_embed_size=obs_embed_size,
                num_tasks=len(self.auxiliary_uuids),
            )

        else:  # single belief model
            state_encoders["single_belief"] = RNNStateEncoder(
                rnn_input_size,
                self._hidden_size,
                num_layers=num_rnn_layers,
                rnn_type=rnn_type,
                trainable_masked_hidden_state=trainable_masked_hidden_state,
            )

        self.state_encoders = nn.ModuleDict(state_encoders)

        self.belief_names = list(self.state_encoders.keys())

        get_logger().info(
            "there are {} belief models: {}".format(
                len(self.belief_names), self.belief_names
            )
        )

    def load_state_dict(self, state_dict, **kwargs):
        new_state_dict = OrderedDict()
        for key in state_dict.keys():
            if "state_encoder." in key:  # old key name
                new_key = key.replace("state_encoder.", "state_encoders.single_belief.")
            elif "goal_visual_encoder.embed_class" in key:
                new_key = key.replace(
                    "goal_visual_encoder.embed_class", "goal_visual_encoder.embed_goal"
                )
            else:
                new_key = key
            new_state_dict[new_key] = state_dict[key]

        return super().load_state_dict(new_state_dict, **kwargs)  # compatible in keys

    def create_actorcritic_head(self):
        self.actor = LinearActorHead(self._hidden_size, self.action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

    def create_aux_models(self, obs_embed_size: int, action_embed_size: int):
        if self.auxiliary_uuids is None:
            return
        aux_models = OrderedDict()
        for aux_uuid in self.auxiliary_uuids:
            aux_models[aux_uuid] = AuxiliaryModel(
                aux_uuid=aux_uuid,
                action_dim=self.action_space.n,
                obs_embed_dim=obs_embed_size,
                belief_dim=self._hidden_size,
                action_embed_size=action_embed_size,
            )

        self.aux_models = nn.ModuleDict(aux_models)

    @property
    def num_recurrent_layers(self):
        """Number of recurrent hidden layers."""
        return list(self.state_encoders.values())[0].num_recurrent_layers

    @property
    def recurrent_hidden_state_size(self):
        """The recurrent hidden state size of a single model."""
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return {
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

    def forward_encoder(
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> torch.FloatTensor:
        raise NotImplementedError("Obs Encoder Not Implemented")

    def fuse_beliefs(
        self, beliefs_dict: Dict[str, torch.FloatTensor], obs_embeds: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        all_beliefs = torch.stack(list(beliefs_dict.values()), dim=-1)  # (T, N, H, k)

        if self.multiple_beliefs:  # call the fusion model
            return self.fusion_model(all_beliefs=all_beliefs, obs_embeds=obs_embeds)
        # single belief
        beliefs = all_beliefs.squeeze(-1)  # (T,N,H)
        return beliefs, None

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
        obs_embeds = self.forward_encoder(observations, memory, prev_actions, masks)

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

        actor_critic_output = ActorCriticOutput(
            distributions=self.actor(beliefs),
            values=self.critic(beliefs),
            extras=extras,
        )

        return actor_critic_output, memory


class SCMBVisualNavActorCritic(ActorCriticModel[CategoricalDistr]):
    """Base class of visual navigation / manipulation (or broadly, embodied AI)
    model.

    `forward_encoder` function requires implementation.
    """

    action_space: gym.spaces.Discrete

    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            hidden_size=512,
            multiple_beliefs=False,
            beliefs_fusion: Optional[FusionType] = None,
            auxiliary_uuids: Optional[List[str]] = None,
            use_transformer_head: bool = False,
            num_decoders: int = 6,
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)
        self._hidden_size = hidden_size
        assert multiple_beliefs == (beliefs_fusion is not None)
        self.multiple_beliefs = multiple_beliefs
        self.beliefs_fusion = beliefs_fusion
        self.auxiliary_uuids = auxiliary_uuids
        if isinstance(self.auxiliary_uuids, list) and len(self.auxiliary_uuids) == 0:
            self.auxiliary_uuids = None

        # Define the placeholders in init function
        self.state_encoders: Optional[nn.ModuleDict] = None
        self.aux_models: Optional[nn.ModuleDict] = None
        self.actor: Optional[LinearActorHead] = None
        self.critic: Optional[LinearCriticHead] = None
        self.prev_action_embedder: Optional[FeatureEmbedding] = None

        self.fusion_model: Optional[nn.Module] = None
        self.belief_names: Optional[Sequence[str]] = None

        self.num_actions = action_space.n
        self.sc_memory_key = ["sc_{}".format(na) for na in range(self.num_actions)]
        self.use_transformer_head = use_transformer_head
        self.transformer_head = None
        self.num_decoders = num_decoders
        self.head_type = None
        self.head_uuid = "{}_{}".format("rnn", "enc")
        self.embed_belief = None

    def create_state_encoders(
            self,
            obs_embed_size: int,
            prev_action_embed_size: int,
            num_rnn_layers: int,
            rnn_type: str,
            add_prev_actions: bool,
            add_prev_action_null_token: bool,
            trainable_masked_hidden_state=False,
    ):
        rnn_input_size = obs_embed_size
        self.prev_action_embedder = FeatureEmbedding(
            input_size=int(add_prev_action_null_token) + self.action_space.n,
            output_size=prev_action_embed_size if add_prev_actions else 0,
        )
        if add_prev_actions:
            rnn_input_size += prev_action_embed_size

        state_encoders = OrderedDict()  # perserve insertion order in py3.6
        if self.multiple_beliefs:  # multiple belief model
            for aux_uuid in self.auxiliary_uuids:
                state_encoders[aux_uuid] = RNNStateEncoder(
                    rnn_input_size,
                    self._hidden_size,
                    num_layers=num_rnn_layers,
                    rnn_type=rnn_type,
                    trainable_masked_hidden_state=trainable_masked_hidden_state,
                )
            # create fusion model
            self.fusion_model = self.beliefs_fusion(
                hidden_size=self._hidden_size,
                obs_embed_size=obs_embed_size,
                num_tasks=len(self.auxiliary_uuids),
            )

        else:  # single belief model
            state_encoders["single_belief"] = RNNStateEncoder(
                rnn_input_size,
                self._hidden_size,
                num_layers=num_rnn_layers,
                rnn_type=rnn_type,
                trainable_masked_hidden_state=trainable_masked_hidden_state,
            )

        self.state_encoders = nn.ModuleDict(state_encoders)

        self.belief_names = list(self.state_encoders.keys())

        get_logger().info(
            "there are {} belief models: {}".format(
                len(self.belief_names), self.belief_names
            )
        )

    def load_state_dict(self, state_dict, **kwargs):
        new_state_dict = OrderedDict()
        for key in state_dict.keys():
            if "state_encoder." in key:  # old key name
                new_key = key.replace("state_encoder.", "state_encoders.single_belief.")
            elif "goal_visual_encoder.embed_class" in key:
                new_key = key.replace(
                    "goal_visual_encoder.embed_class", "goal_visual_encoder.embed_goal"
                )
            else:
                new_key = key
            new_state_dict[new_key] = state_dict[key]

        return super().load_state_dict(new_state_dict, **kwargs)  # compatible in keys

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
            # else:
            #     self.embed_belief = nn.Linear(self._hidden_size // 4, self._hidden_size)



    def create_aux_models(self, obs_embed_size: int, action_embed_size: int):
        if self.auxiliary_uuids is None:
            return
        aux_models = OrderedDict()
        for aux_uuid in self.auxiliary_uuids:
            aux_models[aux_uuid] = AuxiliaryModel(
                aux_uuid=aux_uuid,
                action_dim=self.action_space.n,
                obs_embed_dim=obs_embed_size,
                belief_dim=self._hidden_size,
                action_embed_size=action_embed_size,
            )

        self.aux_models = nn.ModuleDict(aux_models)

    @property
    def num_recurrent_layers(self):
        """Number of recurrent hidden layers."""
        return list(self.state_encoders.values())[0].num_recurrent_layers

    @property
    def recurrent_hidden_state_size(self):
        """The recurrent hidden state size of a single model."""
        return self._hidden_size

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

    def forward_encoder(
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Any, torch.Tensor, Memory]:
        raise NotImplementedError("Obs Encoder Not Implemented")

    def fuse_beliefs(
            self, beliefs_dict: Dict[str, torch.FloatTensor], obs_embeds: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        all_beliefs = torch.stack(list(beliefs_dict.values()), dim=-1)  # (T, N, H, k)

        if self.multiple_beliefs:  # call the fusion model
            return self.fusion_model(all_beliefs=all_beliefs, obs_embeds=obs_embeds)
        # single belief
        beliefs = all_beliefs.squeeze(-1)  # (T,N,H)
        return beliefs, None

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
        obs_embeds, next_state_rep, next_state_pred, memory = self.forward_encoder(observations,
                                                                                   memory,
                                                                                   prev_actions,
                                                                                   masks)

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
        elif not isinstance(self.head, LinearActorCritic):
            representation = next_state_rep
            representation = self.embed_belief(representation)
            beliefs = torch.cat([representation, beliefs.unsqueeze(2)], dim=2)

        actor_critic_output, _ = self.head(
            observations={self.head_uuid: beliefs},
            memory=None,
            prev_actions=prev_actions,
            masks=masks,
        )
        actor_critic_output.extras = extras
        actor_critic_output.extras.update({"model_base_pred": next_state_pred})

        return actor_critic_output, memory
