import json5 as json
import pathlib
import csv
import ast
import numpy as np
from twain_wifco.plant_model import plant_model_from_dict
from twain_wifco.aggregation import aggregation_from_dict
from twain_wifco.metrics_accumulation import metrics_accumulation_from_dict
from twain_wifco.constraint import constraint_from_dict
from twain_wifco.multi_metrics_reduction import multi_metrics_reduction_from_dict
from twain_wifco.optimization import ControlEvaluationSystem

def parse_json_file(path):
    with open(path, "r") as json_file:
        data_dict = json.load(json_file)
        return data_dict
    
def parse_csv_file(csv_path: pathlib.Path,
                   delimiter: str = ";"):
    
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        # Header
        header = next(reader)
        columns = {h_str: [] for h_str in header}

        # Read rows
        for row in reader:
            for h_str, value in zip(header, row):
                columns[h_str].append(np.array(ast.literal_eval(value)))

    data = {}
    for h_str, col in columns.items():
        data[h_str] = np.stack(col, axis=0)

    return data

def control_evaluation_system_from_json(json_path: pathlib.Path):
    
    param_dict = parse_json_file(path=json_path)
    eval_system_name = param_dict["name"]
    config_folder = json_path.parent / pathlib.Path(param_dict["relative_directory"])
    
    # Plant model
    plant_model_dict = parse_json_file(
        path=(config_folder / param_dict["plant_model_file"]))
    plant_model = plant_model_from_dict(param_dict=plant_model_dict)
    
    # Aggregation
    aggregation_dict = parse_json_file(
        path=(config_folder / param_dict["aggregation_file"]))
    aggregation = aggregation_from_dict(param_dict=aggregation_dict)

    # Control constraint
    control_constraint_file = param_dict.get("constraint_control_file", None)
    if control_constraint_file is not None:
        control_constraint_dict = parse_json_file(
            path=(config_folder / control_constraint_file))
        control_constraint = constraint_from_dict(param_dict=control_constraint_dict)
    else:
        control_constraint = None
    
    # Aggregate constraint
    aggregate_constraint_file = param_dict.get("constraint_aggregate_file", None)
    if aggregate_constraint_file is not None:
        aggregate_constraint_dict = parse_json_file(
            path=(config_folder / aggregate_constraint_file))
        aggregate_constraint = constraint_from_dict(param_dict=aggregate_constraint_dict)
    else:
        aggregate_constraint = None

    # Accumulated constraint
    accumulated_constraint_file = param_dict.get("constraint_accumulated_file", None)
    if accumulated_constraint_file is not None:
        accumulated_constraint_dict = parse_json_file(
            path=(config_folder / accumulated_constraint_file))
        accumulated_constraint = constraint_from_dict(param_dict=accumulated_constraint_dict)
    else:
        accumulated_constraint = None

    # Metrics accumulation
    metrics_accumulation_dict = parse_json_file(
        path=config_folder / param_dict["metrics_accumulation_file"])
    metrics_accumulation = metrics_accumulation_from_dict(
        param_dict=metrics_accumulation_dict)

    # Multi-metrics reduction
    multi_metrics_reduction_dict = parse_json_file(
        path=config_folder / param_dict["metrics_reduction_file"])
    multi_metrics_reduction = multi_metrics_reduction_from_dict(
        param_dict=multi_metrics_reduction_dict)
    
    return ControlEvaluationSystem(
        name=eval_system_name,
        plant_model=plant_model,
        aggregation=aggregation,
        constraint_control=control_constraint,
        constraint_aggregate=aggregate_constraint,
        constraint_accumulated=accumulated_constraint,
        metrics_accumulation=metrics_accumulation,
        multi_metrics_reduction=multi_metrics_reduction)
