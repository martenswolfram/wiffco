import os, sys
print(os.getcwd())
import pathlib
from twain_wifco.config import parse_csv_file
from twain_wifco.statistics import statistics_from_table
from twain_wifco.interface import Ambient


data_set = "la_haute_borne"

current_file = pathlib.Path(__file__).resolve()
project_root = current_file.parent.parent

config_dir = project_root / "use_cases" / data_set
csv_path = config_dir / "wind_rose.csv"

statistics_data = parse_csv_file(csv_path=csv_path,
                                 delimiter=";")
support_names={"wind_dir": Ambient.WIND_DIRECTION_DEG,
               "wind_speed_mps": Ambient.WIND_SPEED_MPS}
prevalence_name="prevalence"
                                                    
wind_rose_statistics = statistics_from_table(
    data_dict=statistics_data,
    support_names=support_names,
    prevalence_name=prevalence_name,
    statistics_name="{data_set}_statistics"
)
sys_sample = wind_rose_statistics.systematic_sample(min_prob=0.99)
pass