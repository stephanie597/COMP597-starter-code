from src.config.util.base_config import _Arg, _BaseConfig

config_name="basic_resources"

class TrainerStatsConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_output_dir = _Arg(type=str, help="The path of the output directory where files will be saved.", default=".")
