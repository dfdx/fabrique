from fabrique.loading import LoadConfig
from fabrique.models.phi.model import ModelArgs, Transformer
from fabrique.models.phi.load_rules import RULES


LOAD_CONFIG = LoadConfig(
    model_types=["phi3"],
    model_args_class=ModelArgs,
    model_class=Transformer,
    rules=RULES
)