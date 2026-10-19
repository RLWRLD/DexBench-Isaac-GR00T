# SPDX-License-Identifier: Apache-2.0
"""Modality configuration for RBY1-Wuji2 absolute and arm-relative actions."""

import os

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


action_mode = os.environ.get("RBY1_ACTION_MODE", "abs")
if action_mode not in ("abs", "arm_rel"):
    raise ValueError(f"RBY1_ACTION_MODE must be 'abs' or 'arm_rel', got {action_mode!r}")

groups = [
    "left_arm_joints",
    "right_arm_joints",
    "head_joints",
    "hands_joints",
    "torso_joints",
]
relative_keys = {"left_arm_joints", "right_arm_joints"} if action_mode == "arm_rel" else set()

rby1_wuji2_config = {
    "video": ModalityConfig(
        delta_indices=[0], modality_keys=["camera_ego_left", "camera_ego_right"]
    ),
    "state": ModalityConfig(delta_indices=[0], modality_keys=groups),
    "action": ModalityConfig(
        delta_indices=list(range(40)),
        modality_keys=groups,
        action_configs=[
            ActionConfig(
                rep=(
                    ActionRepresentation.RELATIVE
                    if key in relative_keys
                    else ActionRepresentation.ABSOLUTE
                ),
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for key in groups
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0], modality_keys=["annotation.human.task_description"]
    ),
}

register_modality_config(rby1_wuji2_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
