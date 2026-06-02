# OpenLoong Walking Controller Plugin for genesis-cloud-sim
#
# This plugin provides walking parameter optimization for the OpenLoong
# humanoid robot, adapted from OpenEvolve's openloong_walking example.

__version__ = "0.1.0"
__plugin_name__ = "openloong"
__plugin_type__ = "controller"

# Walking parameters
from .core.walking_params import (
    WalkingParameters,
    WALKING_PRESETS,
    get_preset,
)

# Evaluator
from .core.evaluator import (
    EvaluationResult,
    WalkingEvaluator,
    evaluate_walking,
    compare_parameters,
)

# Environment
from .core.env import (
    OpenLoongWalkingEnv,
    make_env,
)

# Try to import Gymnasium wrapper
try:
    from .core.env import OpenLoongWalkingGymEnv
    __all__ = [
        'WalkingParameters',
        'WALKING_PRESETS',
        'get_preset',
        'EvaluationResult',
        'WalkingEvaluator',
        'evaluate_walking',
        'compare_parameters',
        'OpenLoongWalkingEnv',
        'OpenLoongWalkingGymEnv',
        'make_env',
    ]
except ImportError:
    __all__ = [
        'WalkingParameters',
        'WALKING_PRESETS',
        'get_preset',
        'EvaluationResult',
        'WalkingEvaluator',
        'evaluate_walking',
        'compare_parameters',
        'OpenLoongWalkingEnv',
        'make_env',
    ]
