import os, sys
print(os.getcwd())
import pathlib
from twain_wifco.config import discrete_statistics_from_csv
from twain_wifco.interface import Ambient


data_set = "la_haute_borne"

current_file = pathlib.Path(__file__).resolve()
project_root = current_file.parent.parent

config_dir = project_root / "use_cases" / data_set
csv_path = config_dir / "wind_rose.csv"

wind_rose_statistics = discrete_statistics_from_csv(csv_path=csv_path,
                                                    support_names={"wind_dir": Ambient.WIND_DIRECTION,
                                                                   "wind_speed": Ambient.WIND_SPEED},
                                                    prevalence_name="prevalence",
                                                    delimiter=";",
                                                    statistics_name=f"{data_set}_statistics")

sys_sample = wind_rose_statistics.systematic_sample(min_prob=0.99)