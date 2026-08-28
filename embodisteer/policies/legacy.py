"""Checkpoint-compatibility aliases for policy implementations."""

from .legacy_joint_space import (
    DiffusionUnetTimmPolicyJointSpace as LegacyJointSpacePolicy,
)
from .legacy_joint_space_guidance import (
    DiffusionUnetTimmPolicyJointSpaceWithGuidance as LegacyJointSpaceGuidancePolicy,
)
from .legacy_ee_guidance import (
    DiffusionUnetTimmPolicyWithGuidance as LegacyEEGuidancePolicy,
)

__all__ = [
    "LegacyEEGuidancePolicy",
    "LegacyJointSpacePolicy",
    "LegacyJointSpaceGuidancePolicy",
]
