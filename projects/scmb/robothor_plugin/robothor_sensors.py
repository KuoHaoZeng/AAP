from typing import Any, Tuple, Optional, Union

import gym
import numpy as np
import quaternion  # noqa # pylint: disable=unused-import

from allenact.base_abstractions.sensor import Sensor
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.base_abstractions.task import Task
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from projects.scmb.robothor_plugin.robothor_environment import RoboThorEnvironment
from projects.scmb.robothor_plugin.robothor_tasks import PointNavTask, ObjectNavTask


class RGBSensorRoboThor(RGBSensorThor):
    """Sensor for RGB images in RoboTHOR.

    Returns from a running RoboThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        get_logger().warning(
            "`RGBSensorRoboThor` is deprecated, use `RGBSensorThor` instead."
        )
        super().__init__(*args, **kwargs)

    def frame_from_env(
            self, env: RoboThorEnvironment, task: Optional[PointNavTask]
    ) -> np.ndarray:  # type:ignore
        task.cur_visual_observation = env.current_frame.copy()
        return env.current_frame.copy()


class PrevRGBSensorRoboThor(RGBSensorRoboThor):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def frame_from_env(
            self, env: RoboThorEnvironment, task: Optional[PointNavTask]
    ) -> np.ndarray:  # type:ignore
        if task.prev_visual_observation is None:
            return env.current_frame.copy()
        else:
            return task.prev_visual_observation


class IntermediateRGBSensorRoboThor(RGBSensorRoboThor):
    def __init__(self, index: int = 0, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.index = index

    def frame_from_env(
            self, env: RoboThorEnvironment, task: Optional[ObjectNavTask]
    ) -> np.ndarray:  # type:ignore
        if task.intermediate_visual_observation[self.index] is None:
            return env.current_frame.copy()
        else:
            return task.intermediate_visual_observation[self.index]


class RGBSensorMultiRoboThor(RGBSensor[RoboThorEnvironment, Task[RoboThorEnvironment]]):
    def __init__(self, agent_count: int = 2, **kwargs):
        # TODO take all named args from superclass and pass with super().__init__(**prepare_locals_for_super(locals()))
        super().__init__(**kwargs)
        self.agent_count = agent_count
        self.agent_id = 0

    def frame_from_env(
        self, env: RoboThorEnvironment, task: Optional[Task[RoboThorEnvironment]]
    ) -> np.ndarray:
        return env.current_frames[self.agent_id].copy()

    def get_observation(
        self,
        env: RoboThorEnvironment,
        task: Task[RoboThorEnvironment],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        obs = []
        for self.agent_id in range(self.agent_count):
            obs.append(super().get_observation(env, task, *args, **kwargs))
        return np.stack(obs, axis=0)  # agents x width x height x channels


class GPSCompassSensorRoboThor(Sensor[RoboThorEnvironment, PointNavTask]):
    def __init__(self, uuid: str = "target_coordinates_ind", **kwargs: Any):
        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self):
        return gym.spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(2,),
            dtype=np.float32,
        )

    @staticmethod
    def _compute_pointgoal(
        source_position: np.ndarray,
        source_rotation: np.quaternion,
        goal_position: np.ndarray,
    ):
        direction_vector = goal_position - source_position
        direction_vector_agent = GPSCompassSensorRoboThor.quaternion_rotate_vector(
            source_rotation.inverse(), direction_vector
        )

        rho, phi = GPSCompassSensorRoboThor.cartesian_to_polar(
            direction_vector_agent[2], -direction_vector_agent[0]
        )
        return np.array([rho, phi], dtype=np.float32)

    @staticmethod
    def quaternion_from_y_angle(angle: float) -> np.quaternion:
        r"""Creates a quaternion from rotation angle around y axis"""
        return GPSCompassSensorRoboThor.quaternion_from_coeff(
            np.array(
                [0.0, np.sin(np.pi * angle / 360.0), 0.0, np.cos(np.pi * angle / 360.0)]
            )
        )

    @staticmethod
    def quaternion_from_coeff(coeffs: np.ndarray) -> np.quaternion:
        r"""Creates a quaternions from coeffs in [x, y, z, w] format"""
        quat = np.quaternion(0, 0, 0, 0)
        quat.real = coeffs[3]
        quat.imag = coeffs[0:3]
        return quat

    @staticmethod
    def cartesian_to_polar(x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return rho, phi

    @staticmethod
    def quaternion_rotate_vector(quat: np.quaternion, v: np.array) -> np.array:
        r"""Rotates a vector by a quaternion
        Args:
            quat: The quaternion to rotate by
            v: The vector to rotate
        Returns:
            np.array: The rotated vector
        """
        vq = np.quaternion(0, 0, 0, 0)
        vq.imag = v
        return (quat * vq * quat.inverse()).imag

    def get_observation(
        self,
        env: RoboThorEnvironment,
        task: Optional[PointNavTask],
        *args: Any,
        **kwargs: Any
    ) -> Any:

        agent_state = env.agent_state()
        agent_position = np.array([agent_state[k] for k in ["x", "y", "z"]])
        rotation_world_agent = self.quaternion_from_y_angle(
            agent_state["rotation"]["y"]
        )

        goal_position = np.array([task.task_info["target"][k] for k in ["x", "y", "z"]])

        return self._compute_pointgoal(
            agent_position, rotation_world_agent, goal_position
        )


class AgentStateGPSCompassSensorRoboThor(GPSCompassSensorRoboThor):
    def __init__(
            self,
            uuid: str = "agent_state_target_coordinates_ind",
            use_action_indicator: bool = False,
            **kwargs: Any
    ):
        self.use_action_indicator = use_action_indicator
        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self):
        return gym.spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(6,) if self.use_action_indicator else (5,),
            dtype=np.float32,
        )

    def _compute_state_change_wrt_initial_state(self, agent_position, initial_agent_position, rotation_wrt_initial):
        direction_vector = agent_position - initial_agent_position
        direction_vector_wrt_initial = GPSCompassSensorRoboThor.quaternion_rotate_vector(
            rotation_wrt_initial.inverse(), direction_vector
        )
        return direction_vector_wrt_initial

    def get_agent_state_wrt_initial_state(self, agent_state, initial_agent_state, last_action_success):
        agent_position = np.array([agent_state[k] for k in ["x", "y", "z"]])
        initial_agent_position = np.array([initial_agent_state[k] for k in ["x", "y", "z"]])
        rotation_world_agent_to_initial_agent = self.quaternion_from_y_angle(
            initial_agent_state["rotation"]["y"]
        )
        position_change_wrt_initial = self._compute_state_change_wrt_initial_state(
            agent_position, initial_agent_position, rotation_world_agent_to_initial_agent
        )
        position_change_wrt_initial = [position_change_wrt_initial[0], position_change_wrt_initial[1]]
        rotation_change_wrt_inital = (agent_state["rotation"]["y"] - initial_agent_state["rotation"]["y"]) * np.pi / 180
        if self.use_action_indicator:
            prev_action_success = last_action_success
            return np.hstack([position_change_wrt_initial, rotation_change_wrt_inital, [prev_action_success]])
        else:
            return np.hstack([position_change_wrt_initial, rotation_change_wrt_inital])

    def get_observation(
            self,
            env: RoboThorEnvironment,
            task: Optional[PointNavTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:

        agent_state = env.agent_state()
        agent_position = np.array([agent_state[k] for k in ["x", "y", "z"]])
        rotation_world_agent = self.quaternion_from_y_angle(
            agent_state["rotation"]["y"]
        )

        goal_position = np.array([task.task_info["target"][k] for k in ["x", "y", "z"]])
        pointgoal = self._compute_pointgoal(
            agent_position, rotation_world_agent, goal_position
        )

        initial_agent_state = task.initial_agent_state
        state = self.get_agent_state_wrt_initial_state(agent_state, initial_agent_state, task.last_action_success)

        state = np.hstack([state, pointgoal])
        # Hao: record the state back to the task so the next time can use it
        task.cur_state_observation = np.array(state)

        return state


class PrevAgentStateGPSCompassSensorRoboThor(AgentStateGPSCompassSensorRoboThor):
    def __init__(
            self,
            uuid: str = "prev_agent_state_target_coordinates_ind",
            use_action_indicator: bool = False,
            **kwargs: Any
    ):
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self,
            env: RoboThorEnvironment,
            task: Optional[PointNavTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:

        if task.prev_state_observation is None:
            agent_state = env.agent_state()
            agent_position = np.array([agent_state[k] for k in ["x", "y", "z"]])
            rotation_world_agent = self.quaternion_from_y_angle(
                agent_state["rotation"]["y"]
            )

            goal_position = np.array([task.task_info["target"][k] for k in ["x", "y", "z"]])
            pointgoal = self._compute_pointgoal(
                agent_position, rotation_world_agent, goal_position
            )

            initial_agent_state = task.initial_agent_state
            state = self.get_agent_state_wrt_initial_state(agent_state, initial_agent_state, task.last_action_success)

            state = np.hstack([state, pointgoal])
        else:
            state = task.prev_state_observation

        return state


class DepthSensorThor(
    DepthSensor[
        Union[IThorEnvironment, RoboThorEnvironment],
        Union[Task[IThorEnvironment], Task[RoboThorEnvironment]],
    ],
):
    def __init__(
        self,
        use_resnet_normalization: Optional[bool] = None,
        use_normalization: Optional[bool] = None,
        mean: Optional[np.ndarray] = np.array([[0.5]], dtype=np.float32),
        stdev: Optional[np.ndarray] = np.array([[0.25]], dtype=np.float32),
        height: Optional[int] = None,
        width: Optional[int] = None,
        uuid: str = "depth",
        output_shape: Optional[Tuple[int, ...]] = None,
        output_channels: int = 1,
        unnormalized_infimum: float = 0.0,
        unnormalized_supremum: float = 5.0,
        scale_first: bool = False,
        **kwargs: Any
    ):
        # Give priority to use_normalization, but use_resnet_normalization for backward compat. if not set
        if use_resnet_normalization is not None and use_normalization is None:
            use_normalization = use_resnet_normalization
        elif use_normalization is None:
            use_normalization = False

        super().__init__(**prepare_locals_for_super(locals()))

    def frame_from_env(
        self, env: RoboThorEnvironment, task: Optional[Task[RoboThorEnvironment]]
    ) -> np.ndarray:
        task.cur_depth_observation = env.controller.last_event.depth_frame.copy()
        return env.controller.last_event.depth_frame

class AgentStateSensorRoboThor(GPSCompassSensorRoboThor):
    def __init__(
            self,
            uuid: str = "agent_state",
            **kwargs: Any
    ):
        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self):
        return gym.spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    def _compute_state_change_wrt_initial_state(self, agent_position, initial_agent_position, rotation_wrt_initial):
        direction_vector = agent_position - initial_agent_position
        direction_vector_wrt_initial = GPSCompassSensorRoboThor.quaternion_rotate_vector(
            rotation_wrt_initial.inverse(), direction_vector
        )
        return direction_vector_wrt_initial

    def get_agent_state_wrt_initial_state(self, agent_state, initial_agent_state, last_action_success):
        agent_position = np.array([agent_state[k] for k in ["x", "y", "z"]])
        initial_agent_position = np.array([initial_agent_state[k] for k in ["x", "y", "z"]])
        rotation_world_agent_to_initial_agent = self.quaternion_from_y_angle(
            initial_agent_state["rotation"]["y"]
        )
        position_change_wrt_initial = self._compute_state_change_wrt_initial_state(
            agent_position, initial_agent_position, rotation_world_agent_to_initial_agent
        )
        position_change_wrt_initial = [position_change_wrt_initial[0], position_change_wrt_initial[1]]
        rotation_change_wrt_inital = (agent_state["rotation"]["y"] - initial_agent_state["rotation"]["y"]) * np.pi / 180
        return np.hstack([position_change_wrt_initial, rotation_change_wrt_inital])

    def get_observation(
            self,
            env: RoboThorEnvironment,
            task: Optional[ObjectNavTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:

        agent_state = env.agent_state()
        initial_agent_state = task.initial_agent_state
        state = self.get_agent_state_wrt_initial_state(agent_state, initial_agent_state, task.last_action_success)

        # Hao: record the state back to the task so the next time can use it
        task.cur_state_observation = np.array(state)

        return state


class PrevAgentStateSensorRoboThor(AgentStateSensorRoboThor):
    def __init__(
            self,
            uuid: str = "prev_agent_state",
            **kwargs: Any
    ):
        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self):
        return gym.spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    def get_observation(
            self,
            env: RoboThorEnvironment,
            task: Optional[ObjectNavTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:

        if task.prev_state_observation is None:
            agent_state = env.agent_state()
            initial_agent_state = task.initial_agent_state
            state = self.get_agent_state_wrt_initial_state(agent_state, initial_agent_state, task.last_action_success)
        else:
            state = task.prev_state_observation

        return state
