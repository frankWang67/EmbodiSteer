"""Training-side adapters owned by the EmbodiSteer release."""

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class NoOpImageRunner(BaseImageRunner):
    """Disable in-training rollouts; simulation evaluation is a separate stage."""

    def run(self, policy: BaseImagePolicy) -> dict:
        return {}


__all__ = ["NoOpImageRunner"]
