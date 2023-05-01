import torch

from allenact.utils.experiment_utils import Builder
from allenact.utils.experiment_utils import TrainingPipeline
from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from projects.scmb.pointnav.robothor.pointnav_robothor_base import (
    PointNavRoboThorBaseConfig,
)
from projects.scmb.pointnav.resnet_mixins import (
    PointNavResNetWithGRUActorCriticMixin,
    PointNavPPOMixin
)
from projects.scmb.robothor_plugin.robothor_sensors import AgentStateGPSCompassSensorRoboThor


class PointNavRoboThorRGBPPOExperimentConfig(PointNavRoboThorBaseConfig,):
    """An Point Navigation experiment configuration in RoboThor with RGB
    input."""

    SENSORS = [
        RGBSensorThor(
            height=PointNavRoboThorBaseConfig.SCREEN_SIZE,
            width=PointNavRoboThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
            uuid="rgb_lowres",
        ),
        AgentStateGPSCompassSensorRoboThor(),
    ]

    def __init__(self):
        super().__init__()

        self.model_creation_handler = PointNavResNetWithGRUActorCriticMixin(
            backbone="resnet",
            sensors=self.SENSORS,
            auxiliary_uuids=[],
            add_prev_actions=True,
            multiple_beliefs=False,
            belief_fusion=None,
        )

    def preprocessors(self):
        return [
            Builder(
                ClipResNetPreprocessor,
                {
                    "rgb_input_uuid": "rgb_lowres",
                    "output_uuid": "rgb_resnet",
                    "clip_model_type": "RN50",
                    "pool": True,
                    "device": torch.device("cuda"),
                    "device_ids": list(range(torch.cuda.device_count())),
                }
            )
        ]

    def training_pipeline(self, **kwargs) -> TrainingPipeline:
        return PointNavPPOMixin.training_pipeline(
            auxiliary_uuids=[],
            multiple_beliefs=False,
            normalize_advantage=True,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
        )

    def create_model(self, **kwargs):
        return self.model_creation_handler.create_model(**kwargs)

    def tag(self):
        return "PointNav-RoboTHOR-RGB-ResNet-EmbCLIP-DDPPO"
