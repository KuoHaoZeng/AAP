import os
from abc import ABC

from projects.scmb.pointnav.pointnav_thor_base import (
    PointNavThorBaseConfig,
)


class PointNavRoboThorBaseConfig(PointNavThorBaseConfig, ABC):
    """The base config for all iTHOR PointNav pointnav."""

    NUM_PROCESSES = 48

    TRAIN_DATASET_DIR = os.path.join(os.getcwd(), "datasets/robothor-pointnav/train")
    VAL_DATASET_DIR = os.path.join(os.getcwd(), "datasets/robothor-pointnav/val")
    # VAL_DATASET_DIR = os.path.join(os.getcwd(), "datasets/robothor-pointnav/ArchitecTHOR_val")
    TRAIN_ROTATION_BIAS = [-15, 15]
    VAL_ROTATION_BIAS = [30, 330]
    TEST_ROTATION_BIAS = [90, 90]

    TRAIN_MOVEMENT_BIAS = [-0.05, 0.05]
    VAL_MOVEMENT_BIAS = [-0.2, 0.2, -0.1, 0.1]
    TEST_MOVEMENT_BIAS = [0.2, 0.2]

    TEST_NUM_BROKEN_ROTATION_ACTIONS = 0

    TEST_CONTEXT_BROKEN = True
