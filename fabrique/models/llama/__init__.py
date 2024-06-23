from fabrique.loading import LoadConfig
from fabrique.models.llama.model import ModelArgs, Transformer
from fabrique.models.llama.load_rules import RULES


LOAD_CONFIG = LoadConfig(
    model_types=["llama"],
    model_args_class=ModelArgs,
    model_class=Transformer,
    rules=RULES
)