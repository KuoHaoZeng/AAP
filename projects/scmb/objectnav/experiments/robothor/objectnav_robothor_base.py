import os
from abc import ABC
from typing import Optional, List, Any, Dict

import torch

from allenact.utils.misc_utils import prepare_locals_for_super
from projects.scmb.objectnav.experiments.objectnav_thor_base import (
    ObjectNavThorBaseConfig,
)


class ObjectNavRoboThorBaseConfig(ObjectNavThorBaseConfig, ABC):
    """The base config for all RoboTHOR ObjectNav pointnav."""
    THOR_COMMIT_ID = "bad5bc2b250615cb766ffb45d455c211329af17e"
    THOR_COMMIT_ID_FOR_RAND_MATERIALS = "9549791ce2e7f472063a10abb1fb7664159fec23"

    AGENT_MODE = "locobot"

    DEFAULT_NUM_TRAIN_PROCESSES = 48 if torch.cuda.is_available() else 1

    TRAIN_DATASET_DIR = os.path.join(os.getcwd(), "datasets/robothor-objectnav/train")
    VAL_DATASET_DIR = os.path.join(os.getcwd(), "datasets/robothor-objectnav/val")
    TEST_DATASET_DIR = os.path.join(os.getcwd(), "datasets/robothor-objectnav/test")

    TRAIN_ROTATION_BIAS = [-15, 15]
    VAL_ROTATION_BIAS = [30, 330]
    TEST_ROTATION_BIAS = [90, 90]

    TRAIN_MOVEMENT_BIAS = [-0.05, 0.05]
    VAL_MOVEMENT_BIAS = [-0.2, 0.2, -0.1, 0.1]
    TEST_MOVEMENT_BIAS = [0.2, 0.2]

    TEST_NUM_BROKEN_ROTATION_ACTIONS = 0

    TEST_CONTEXT_BROKEN = True

    TARGET_TYPES = tuple(
        sorted(
            [
                "AlarmClock",
                "Apple",
                "BaseballBat",
                "BasketBall",
                "Bowl",
                "GarbageCan",
                "HousePlant",
                "Laptop",
                "Mug",
                "SprayBottle",
                "Television",
                "Vase",
            ]
        )
    )

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        kwargs = super(ObjectNavRoboThorBaseConfig, self).train_task_sampler_args(
            **prepare_locals_for_super(locals())
        )
        if self.randomize_train_materials:
            kwargs["env_args"]["commit_id"] = self.THOR_COMMIT_ID_FOR_RAND_MATERIALS
        return kwargs
