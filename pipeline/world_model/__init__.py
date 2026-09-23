"""Action-conditioned world model: predict an action's consequences before acting.

Explicit `__init__.py` rather than relying on PEP 420 namespace packages. An implicit
namespace package imports successfully even when nothing is there, which is how
`import mycpp` silently returned an empty module in `foundationpose_6dof` and defeated
the try/except guard meant to catch its absence. A missing package should raise.
"""

from pipeline.world_model.latent import SceneLatent
from pipeline.world_model.dynamics import DynamicsEnsemble, LatentDynamics
from pipeline.world_model.planner import ActionPlanner, PlanResult, perturbed_candidates
from pipeline.world_model.scorer import (ScoreBreakdown, ScoreWeights, TrajectoryScorer,
                                         ValueHead)

__all__ = ["SceneLatent", "LatentDynamics", "DynamicsEnsemble", "ActionPlanner",
           "PlanResult", "perturbed_candidates", "TrajectoryScorer", "ScoreWeights",
           "ScoreBreakdown", "ValueHead"]
