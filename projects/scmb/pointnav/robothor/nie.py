from allenact.utils.experiment_utils import TrainingPipeline
from projects.scmb.robothor_plugin.robothor_sensors import (
    AgentStateGPSCompassSensorRoboThor,
    RGBSensorRoboThor,
    PrevAgentStateGPSCompassSensorRoboThor,
    PrevRGBSensorRoboThor,
)
from projects.scmb.pointnav.scmb_resnet_mixins import (
    PointNavResNetWithGRUActorCriticMixin,
    PointNavPPOMixin
)
from projects.scmb.pointnav.robothor.pointnav_robothor_base import (
    PointNavRoboThorBaseConfig,
)
from allenact.utils.experiment_utils import Builder
from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor
import torch


class NIE_PointNavRoboThorRGBPPOExperimentConfig(PointNavRoboThorBaseConfig,):
    """An Point Navigation experiment configuration in RoboThor with RGB
    input."""

    CERTAIN_ACTION_STATE_CHANGE_EMBED = [15]
    TEST_VISUALIZE_OR_NOT = False

    SENSORS = [
        RGBSensorRoboThor(
            height=PointNavRoboThorBaseConfig.SCREEN_SIZE,
            width=PointNavRoboThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
            uuid="rgb_lowres",
        ),
        AgentStateGPSCompassSensorRoboThor(),
        PrevRGBSensorRoboThor(
            height=PointNavRoboThorBaseConfig.SCREEN_SIZE,
            width=PointNavRoboThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
            uuid="prev_rgb_lowres",
        ),
        PrevAgentStateGPSCompassSensorRoboThor(),
    ]

    def __init__(self):
        super().__init__()

        self.model_creation_handler = PointNavResNetWithGRUActorCriticMixin(
            backbone="resnet",
            sensors=self.SENSORS,
            auxiliary_uuids=[],
            add_prev_actions=False,
            multiple_beliefs=False,
            belief_fusion=None,
            certain_action_state_change_embed=self.CERTAIN_ACTION_STATE_CHANGE_EMBED,
            use_transformer_head=False,
            num_decoders=6,
            head_type="Linear",
            is_vo_baseline=True,
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
            ),
            Builder(
                ClipResNetPreprocessor,
                {
                    "rgb_input_uuid": "prev_rgb_lowres",
                    "output_uuid": "prev_rgb_resnet",
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
            certain_action_state_change_embed=self.CERTAIN_ACTION_STATE_CHANGE_EMBED,
        )

    def create_model(self, **kwargs):
        return self.model_creation_handler.create_model(**kwargs)

    def tag(self):
        return "PointNav-RoboTHOR-RGB-ResNet-NIE-DDPPO"
