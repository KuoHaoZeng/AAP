from typing import Sequence, Union

import torch.nn as nn

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.experiment_utils import Builder, TrainingPipeline
from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor
from allenact_plugins.ithor_plugin.ithor_sensors import (
    GoalObjectTypeThorSensor,
)
from projects.scmb.objectnav.experiments.clip.scmb_mixins import (
    SCMBClipResNetPreprocessGRUActorCriticMixin
)
from projects.scmb.objectnav.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)
from projects.scmb.objectnav.mb_mixins import ObjectNavMBPPOMixin
from projects.scmb.robothor_plugin.robothor_sensors import (
    RGBSensorRoboThor,
    PrevRGBSensorRoboThor,
    AgentStateSensorRoboThor,
    IntermediateRGBSensorRoboThor,
)


class ObjectNavRoboThorClipRGBPPOExperimentConfig(ObjectNavRoboThorBaseConfig):
    """An Object Navigation experiment configuration in RoboThor with RGB
    input."""
    STATE_CHANGES_PRED_DIMS = 3
    CERTAIN_ACTION_STATE_CHANGE_EMBED = [15]
    NUM_INTERMEDIATE_VISUAL_OBSERVATION = 1
    TEST_VISUALIZE_OR_NOT = False

    CLIP_MODEL_TYPE = "RN50"

    SENSORS = [
        RGBSensorRoboThor(
            height=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            width=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
        ),
        PrevRGBSensorRoboThor(
            height=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            width=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="prev_rgb_lowres",
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
        ),
        GoalObjectTypeThorSensor(
            object_types=ObjectNavRoboThorBaseConfig.TARGET_TYPES,
        ),
        AgentStateSensorRoboThor(),
        IntermediateRGBSensorRoboThor(
            index=0,
            height=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            width=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="inter0_rgb_lowres",
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
        )
    ]

    def __init__(self, add_prev_actions: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.preprocessing_and_model = SCMBClipResNetPreprocessGRUActorCriticMixin(
            sensors=self.SENSORS,
            clip_model_type=self.CLIP_MODEL_TYPE,
            screen_size=self.SCREEN_SIZE,
            goal_sensor_type=GoalObjectTypeThorSensor,
            state_changes_pred_dims=self.STATE_CHANGES_PRED_DIMS,
            certain_action_state_change_embed=self.CERTAIN_ACTION_STATE_CHANGE_EMBED,
            use_transformer_head=True,
            num_decoders=6,
            head_type="OrderInvariant",
        )
        self.add_prev_actions = add_prev_actions

    def training_pipeline(self, **kwargs) -> TrainingPipeline:
        return ObjectNavMBPPOMixin.training_pipeline(
            auxiliary_uuids=[],
            multiple_beliefs=False,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
            state_changes_pred_dims=self.STATE_CHANGES_PRED_DIMS,
            certain_action_state_change_embed=self.CERTAIN_ACTION_STATE_CHANGE_EMBED,
        )

    def preprocessors(self) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        return self.preprocessing_and_model.preprocessors()

    def create_model(self, **kwargs) -> nn.Module:
        return self.preprocessing_and_model.create_model(
            num_actions=self.ACTION_SPACE.n, add_prev_actions=self.add_prev_actions, **kwargs
        )

    @classmethod
    def tag(cls):
        return "ObjectNav-RoboTHOR-RGB-ClipResNet50GRU-AAP-DDPPO"
