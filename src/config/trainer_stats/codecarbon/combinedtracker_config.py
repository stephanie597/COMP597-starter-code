"""Configuration for Combined Stats (Resource + Carbon).

This config file defines parameters for combined monitoring
of system resources and carbon emissions.
"""

from src.config.util.base_config import _Arg, _BaseConfig

config_name = "combined_stats"


class TrainerStatsConfig(_BaseConfig):
    """Configuration for combined resource and carbon tracking.
    
    Attributes
    ----------
    run_num : int
        Run number for tracking different experiments.
        Default: 1
    
    project_name : str
        Project name for tracking and identification.
        Default: "resnet152_combined"
    
    output_dir : str
        Directory where all output files will be saved.
        Default: "./training_stats_backup"
    """
    
    def __init__(self) -> None:
        super().__init__()
        
        self._arg_run_num = _Arg(
            type=int,
            help="Run number for tracking different experiments.",
            default=1
        )
        
        self._arg_project_name = _Arg(
            type=str,
            help="Project name for tracking and identification.",
            default="resnet152_combined"
        )
        
        self._arg_output_dir = _Arg(
            type=str,
            help="Output directory for all stats files.",
            default="./training_stats_backup"
        )