from fabrique.loading import LoadConfig
from fabrique.models.phi.load_rules import RULES
from fabrique.models.phi.modeling import ModelArgs, Transformer

LOAD_CONFIG = LoadConfig(
    model_types=["phi3"],
    model_args_class=ModelArgs,
    model_class=Transformer,
    rules=RULES,
)
