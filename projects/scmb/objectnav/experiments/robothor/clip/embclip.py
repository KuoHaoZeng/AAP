from typing import Sequence, Union

import torch.nn as nn

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.experiment_utils import Builder, TrainingPipeline
from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor
from allenact_plugins.ithor_plugin.ithor_sensors import (
    GoalObjectTypeThorSensor,
    RGBSensorThor,
)
from projects.scmb.objectnav.experiments.clip.mixins import (
    ClipResNetPreprocessGRUActorCriticMixin,
)
from projects.scmb.objectnav.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)
from projects.scmb.objectnav.mixins import ObjectNavPPOMixin
from projects.scmb.robothor_plugin.robothor_sensors import IntermediateRGBSensorRoboThor


class ObjectNavRoboThorClipRGBPPOExperimentConfig(ObjectNavRoboThorBaseConfig):
    """An Object Navigation experiment configuration in RoboThor with RGB
    input."""

    CLIP_MODEL_TYPE = "RN50"
    NUM_INTERMEDIATE_VISUAL_OBSERVATION = 1

    SENSORS = [
        RGBSensorThor(
            height=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            width=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
        ),
        GoalObjectTypeThorSensor(
            object_types=ObjectNavRoboThorBaseConfig.TARGET_TYPES,
        ),
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

        self.preprocessing_and_model = ClipResNetPreprocessGRUActorCriticMixin(
            sensors=self.SENSORS,
            clip_model_type=self.CLIP_MODEL_TYPE,
            screen_size=self.SCREEN_SIZE,
            goal_sensor_type=GoalObjectTypeThorSensor,
        )
        self.add_prev_actions = add_prev_actions

    def training_pipeline(self, **kwargs) -> TrainingPipeline:
        return ObjectNavPPOMixin.training_pipeline(
            auxiliary_uuids=[],
            multiple_beliefs=False,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
        )

    def preprocessors(self) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        return self.preprocessing_and_model.preprocessors()

    def create_model(self, **kwargs) -> nn.Module:
        return self.preprocessing_and_model.create_model(
            num_actions=self.ACTION_SPACE.n, add_prev_actions=self.add_prev_actions, **kwargs
        )

    @classmethod
    def tag(cls):
        return "ObjectNav-RoboTHOR-RGB-ClipResNet50GRU-EmbCLIP-DDPPO"
