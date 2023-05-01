from typing import Sequence, Union, Type

import attr
import gym
import numpy as np
import torch.nn as nn

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.base_abstractions.sensor import Sensor
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.utils.experiment_utils import Builder
from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor
from projects.scmb.navigation_plugin.objectnav.models import (
    SCMBResnetTensorNavActorCritic,
)
from projects.scmb.robothor_plugin.robothor_sensors import PrevRGBSensorRoboThor, IntermediateRGBSensorRoboThor
from projects.scmb.models.basic_models import LinearActorCritic, OrderInvariantActorCritic


@attr.s(kw_only=True)
class SCMBClipResNetPreprocessGRUActorCriticMixin:
    sensors: Sequence[Sensor] = attr.ib()
    clip_model_type: str = attr.ib()
    screen_size: int = attr.ib()
    goal_sensor_type: Type[Sensor] = attr.ib()
    pool: bool = attr.ib(default=False)
    state_changes_pred_dims: int = attr.ib()
    use_transformer_head: bool = attr.ib()
    num_decoders: int = attr.ib()
    head_type: str = attr.ib()
    certain_action_state_change_embed: Sequence[int] = attr.ib(default=[])
    is_vo_baseline: bool = attr.ib(default=False)

    def preprocessors(self) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        preprocessors = []

        rgb_sensor = next((s for s in self.sensors if isinstance(s, RGBSensor)), None)
        assert (
            np.linalg.norm(
                np.array(rgb_sensor._norm_means)
                - np.array(ClipResNetPreprocessor.CLIP_RGB_MEANS)
            )
            < 1e-5
        )
        assert (
            np.linalg.norm(
                np.array(rgb_sensor._norm_sds)
                - np.array(ClipResNetPreprocessor.CLIP_RGB_STDS)
            )
            < 1e-5
        )

        if rgb_sensor is not None:
            preprocessors.append(
                ClipResNetPreprocessor(
                    rgb_input_uuid=rgb_sensor.uuid,
                    clip_model_type=self.clip_model_type,
                    pool=self.pool,
                    output_uuid="rgb_clip_resnet",
                )
            )

        prev_rgb_sensor = next((s for s in self.sensors if isinstance(s, PrevRGBSensorRoboThor)), None)
        assert (
                np.linalg.norm(
                    np.array(prev_rgb_sensor._norm_means)
                    - np.array(ClipResNetPreprocessor.CLIP_RGB_MEANS)
                )
                < 1e-5
        )
        assert (
                np.linalg.norm(
                    np.array(prev_rgb_sensor._norm_sds)
                    - np.array(ClipResNetPreprocessor.CLIP_RGB_STDS)
                )
                < 1e-5
        )

        if prev_rgb_sensor is not None:
            preprocessors.append(
                ClipResNetPreprocessor(
                    rgb_input_uuid=prev_rgb_sensor.uuid,
                    clip_model_type=self.clip_model_type,
                    pool=self.pool,
                    output_uuid="prev_rgb_clip_resnet",
                )
            )

        inter0_rgb_sensor = next((s for s in self.sensors if isinstance(s, IntermediateRGBSensorRoboThor) and "inter0" in s.uuid), None)

        if inter0_rgb_sensor is not None:
            assert (
                    np.linalg.norm(
                        np.array(inter0_rgb_sensor._norm_means)
                        - np.array(ClipResNetPreprocessor.CLIP_RGB_MEANS)
                    )
                    < 1e-5
            )
            assert (
                    np.linalg.norm(
                        np.array(inter0_rgb_sensor._norm_sds)
                        - np.array(ClipResNetPreprocessor.CLIP_RGB_STDS)
                    )
                    < 1e-5
            )
            preprocessors.append(
                ClipResNetPreprocessor(
                    rgb_input_uuid=inter0_rgb_sensor.uuid,
                    clip_model_type=self.clip_model_type,
                    pool=self.pool,
                    output_uuid="inter0_rgb_clip_resnet",
                )
            )

        inter1_rgb_sensor = next((s for s in self.sensors if isinstance(s, IntermediateRGBSensorRoboThor) and "inter1" in s.uuid), None)

        if inter1_rgb_sensor is not None:
            assert (
                    np.linalg.norm(
                        np.array(inter1_rgb_sensor._norm_means)
                        - np.array(ClipResNetPreprocessor.CLIP_RGB_MEANS)
                    )
                    < 1e-5
            )
            assert (
                    np.linalg.norm(
                        np.array(inter1_rgb_sensor._norm_sds)
                        - np.array(ClipResNetPreprocessor.CLIP_RGB_STDS)
                    )
                    < 1e-5
            )
            preprocessors.append(
                ClipResNetPreprocessor(
                    rgb_input_uuid=inter1_rgb_sensor.uuid,
                    clip_model_type=self.clip_model_type,
                    pool=self.pool,
                    output_uuid="inter1_rgb_clip_resnet",
                )
            )

        inter2_rgb_sensor = next((s for s in self.sensors if isinstance(s, IntermediateRGBSensorRoboThor) and "inter2" in s.uuid), None)

        if inter2_rgb_sensor is not None:
            assert (
                    np.linalg.norm(
                        np.array(inter2_rgb_sensor._norm_means)
                        - np.array(ClipResNetPreprocessor.CLIP_RGB_MEANS)
                    )
                    < 1e-5
            )
            assert (
                    np.linalg.norm(
                        np.array(inter2_rgb_sensor._norm_sds)
                        - np.array(ClipResNetPreprocessor.CLIP_RGB_STDS)
                    )
                    < 1e-5
            )
            preprocessors.append(
                ClipResNetPreprocessor(
                    rgb_input_uuid=inter2_rgb_sensor.uuid,
                    clip_model_type=self.clip_model_type,
                    pool=self.pool,
                    output_uuid="inter2_rgb_clip_resnet",
                )
            )

        inter3_rgb_sensor = next((s for s in self.sensors if isinstance(s, IntermediateRGBSensorRoboThor) and "inter3" in s.uuid), None)

        if inter3_rgb_sensor is not None:
            assert (
                    np.linalg.norm(
                        np.array(inter3_rgb_sensor._norm_means)
                        - np.array(ClipResNetPreprocessor.CLIP_RGB_MEANS)
                    )
                    < 1e-5
            )
            assert (
                    np.linalg.norm(
                        np.array(inter3_rgb_sensor._norm_sds)
                        - np.array(ClipResNetPreprocessor.CLIP_RGB_STDS)
                    )
                    < 1e-5
            )
            preprocessors.append(
                ClipResNetPreprocessor(
                    rgb_input_uuid=inter3_rgb_sensor.uuid,
                    clip_model_type=self.clip_model_type,
                    pool=self.pool,
                    output_uuid="inter3_rgb_clip_resnet",
                )
            )

        depth_sensor = next(
            (s for s in self.sensors if isinstance(s, DepthSensor)), None
        )
        if depth_sensor is not None:
            preprocessors.append(
                ClipResNetPreprocessor(
                    rgb_input_uuid=depth_sensor.uuid,
                    clip_model_type=self.clip_model_type,
                    pool=self.pool,
                    output_uuid="depth_clip_resnet",
                )
            )

        return preprocessors

    def create_model(self, num_actions: int, add_prev_actions: bool, **kwargs) -> nn.Module:
        has_rgb = any(isinstance(s, RGBSensor) for s in self.sensors)
        has_depth = any(isinstance(s, DepthSensor) for s in self.sensors)
        has_prev_rgb = any(isinstance(s, PrevRGBSensorRoboThor) for s in self.sensors)
        has_inter0_rgb = any(isinstance(s, IntermediateRGBSensorRoboThor) and "inter0" in s.uuid for s in self.sensors)
        has_inter1_rgb = any(isinstance(s, IntermediateRGBSensorRoboThor) and "inter1" in s.uuid for s in self.sensors)
        has_inter2_rgb = any(isinstance(s, IntermediateRGBSensorRoboThor) and "inter2" in s.uuid for s in self.sensors)
        has_inter3_rgb = any(isinstance(s, IntermediateRGBSensorRoboThor) and "inter3" in s.uuid for s in self.sensors)

        goal_sensor_uuid = next(
            (s.uuid for s in self.sensors if isinstance(s, self.goal_sensor_type)),
            None,
        )

        if self.use_transformer_head:
            head_type = OrderInvariantActorCritic
        else:
            if self.head_type == "OrderInvariant":
                head_type = OrderInvariantActorCritic
            else:
                head_type = LinearActorCritic

        return SCMBResnetTensorNavActorCritic(
            action_space=gym.spaces.Discrete(num_actions),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            rgb_resnet_preprocessor_uuid="rgb_clip_resnet" if has_rgb else None,
            depth_resnet_preprocessor_uuid="depth_clip_resnet" if has_depth else None,
            prev_rgb_resnet_preprocessor_uuid="prev_rgb_clip_resnet" if has_prev_rgb else None,
            inter0_rgb_resnet_preprocessor_uuid="inter0_rgb_clip_resnet" if has_inter0_rgb else None,
            inter1_rgb_resnet_preprocessor_uuid="inter1_rgb_clip_resnet" if has_inter1_rgb else None,
            inter2_rgb_resnet_preprocessor_uuid="inter2_rgb_clip_resnet" if has_inter2_rgb else None,
            inter3_rgb_resnet_preprocessor_uuid="inter3_rgb_clip_resnet" if has_inter3_rgb else None,
            hidden_size=512,
            goal_dims=32,
            add_prev_actions=add_prev_actions,
            state_changes_pred_dims=self.state_changes_pred_dims,
            certain_action_state_change_embed=self.certain_action_state_change_embed,
            use_transformer_head=self.use_transformer_head,
            num_decoders=self.num_decoders,
            head_type=head_type,
            is_vo_baseline=self.is_vo_baseline,
        )
