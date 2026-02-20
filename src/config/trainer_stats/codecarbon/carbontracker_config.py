"""Configuration for Carbon Tracker.

This config file defines parameters for tracking carbon emissions
during model training using CodeCarbon.
"""

from src.config.util.base_config import _Arg, _BaseConfig

# If placing directly in trainer_stats/ without subdirectory
config_name = "carbon_tracker_config"


class TrainerStatsConfig(_BaseConfig):
    """Configuration for carbon emissions tracking.
    
    Attributes
    ----------
    run_num : int
        Run number for tracking different experiments.
        Used in output filenames to distinguish runs.
        Default: 1
    
    project_name : str
        Name of the project for CodeCarbon tracking.
        Used to identify the experiment.
        Default: "resnet152_carbon"
    
    output_dir : str
        Directory where carbon tracking CSV files will be saved.
        Default: "./training_stats_backup"
    """
    
    def __init__(self) -> None:
        super().__init__()
        
        self._arg_run_num = _Arg(
            type=int,
            help="Run number for tracking different experiments (used in filenames).",
            default=1
        )
        
        self._arg_project_name = _Arg(
            type=str,
            help="Project name for CodeCarbon tracking and identification.",
            default="resnet152_carbon"
        )
        
        self._arg_output_dir = _Arg(
            type=str,
            help="Output directory for saving carbon emission CSV files.",
            default="./training_stats_backup"
        )