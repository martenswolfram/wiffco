from typing import Dict, Any, Tuple, Sequence
from abc import ABC, abstractmethod
import numpy as np
import logging
from scipy.optimize import minimize, Bounds
from scipy.sparse import coo_matrix
from functools import lru_cache
import itertools
from enum import Enum
from twain_wifco.interface import (
    Ambient,
    Control,
    DataTable,
    DataTable)
from twain_wifco.statistics import (
    AmbientStatistics,
    DiscreteAmbientStatistics)
from twain_wifco.control_policy import (
    DiscreteControlPolicy,
    default_discrete_policy
    )
from twain_wifco.plant_model import PlantModel
from twain_wifco.aggregation import Aggregation
from twain_wifco.metrics_accumulation import MetricsAccumulation
from twain_wifco.constraint import ControlConstraint, AggregateConstraint, AccumulatedConstraint
from twain_wifco.multi_metrics_reduction import MultiMetricsReduction

logger = logging.getLogger(__name__)

MAX_NUM_EVALS = 1e4
MAX_NUM_POLICIES = 1e7

class ControlEvaluationSystem:
    def __init__(self,
                 name: str, 
                 plant_model: PlantModel,
                 aggregation: Aggregation,
                 constraint_control: ControlConstraint | None,
                 constraint_aggregate: AggregateConstraint | None,
                 constraint_accumulated: AccumulatedConstraint | None,
                 metrics_accumulation: MetricsAccumulation,
                 multi_metrics_reduction: MultiMetricsReduction):
        
        # Parameters
        self.name = name
        self.plant_model = plant_model
        self.aggregation = aggregation
        self.constraint_control = constraint_control
        self.constraint_aggregate = constraint_aggregate
        self.constraint_accumulated = constraint_accumulated
        self.metrics_accumulation = metrics_accumulation
        self.multi_metrics_reduction = multi_metrics_reduction

    def aggregate_from_amb(self,
                           ambient: DataTable[Ambient],
                           control: DataTable[Control]):
        
        model_output = self.plant_model.evaluate(
             meteorological=ambient,
             control=control)
        return self.aggregation.compute_aggregate(model_output=model_output,
                                                  ambient=ambient,
                                                  control=control)


    def aggregate_from_amb_constr_eval(self,
                                       ambient: DataTable[Ambient],
                                       control: DataTable[Control]):
        
        # Indices that are admissible w.r.t. to control
        control_admissible = np.arange(len(ambient))
        if self.constraint_control is not None:
            control_admissible = self.constraint_control.evaluate_satisfied(
                constraint_input=control)
        if len(control_admissible) == 0:
            return None, np.array([])
        
        # Compute aggregates for admissible controls only
        control_admissible_aggregates = self.aggregate_from_amb(
            ambient=ambient.extract(indices=control_admissible),
            control=control.extract(indices=control_admissible))
        
        # Aggregate-admissible indices among control-admissible
        aggr_admissible_among_ctrl_admissible = np.arange(len(control_admissible))
        if self.constraint_aggregate is not None:
            aggr_admissible_among_ctrl_admissible = self.constraint_aggregate.evaluate_satisfied(
                constraint_input=control_admissible_aggregates)

        # Both control and aggregate admissible indices
        admissible = control_admissible[aggr_admissible_among_ctrl_admissible]
        if len(admissible) == 0:
            return None, np.array([])
        
        # Return admissible aggregates and indices
        admissible_aggregates = control_admissible_aggregates.extract(indices=aggr_admissible_among_ctrl_admissible)
        return admissible_aggregates, admissible
                
    def acc_metrics_from_amb(self,
                             ambient: DataTable[Ambient],
                             control: DataTable[Control]):
        
        aggregate = self.aggregate_from_amb(
            ambient=ambient,
            control=control)

        acc_metrics = self.metrics_accumulation.acc_metrics(
            aggregate=aggregate)
            
        return acc_metrics
    
    def acc_metrics_from_amb_constr_eval(self,
                                         ambient: DataTable[Ambient],
                                         control: DataTable[Control]):
        
        aggregate, constraints_satisfied = self.aggregate_from_amb_constr_eval(
            ambient=ambient,
            control=control)
        
        if not constraints_satisfied:
            return None, False
        
        acc_metrics = self.metrics_accumulation.acc_metrics(
            aggregate=aggregate)
        
        if self.constraint_accumulated is not None:
            if not self.constraint_accumulated.evaluate_satisfied(
                constraint_input=acc_metrics):
                return None, False
            
        return acc_metrics, True
    
    def expected_acc_metrics(self,
                             ambient_statistics: AmbientStatistics,
                             control_policy: DiscreteControlPolicy,
                             max_num_amb_cond: int | None = None):
        
        ambient_sample = ambient_statistics.systematic_sample(N_max=max_num_amb_cond)
        ambient = ambient_sample.ambient_support
        control = control_policy.get_control(ambient=ambient)
        aggregate = self.aggregate_from_amb(
            ambient=ambient,
            control=control)
        acc_metrics = self.metrics_accumulation.acc_metrics(aggregate=aggregate)
        return acc_metrics.expected_value(
            probabilities=ambient_sample.normalized_weights[:, np.newaxis])

    def expected_acc_metrics_constr_eval(self,
                                         ambient_statistics: AmbientStatistics,
                                         control_policy: DiscreteControlPolicy,
                                         max_num_amb_cond: int | None = None):
        
        ambient_sample = ambient_statistics.systematic_sample(N_max=max_num_amb_cond)
        ambient = ambient_sample.ambient_support
        control = control_policy.get_control(ambient=ambient)
        aggregate, constraints_satisfied = self.aggregate_from_amb_constr_eval(
            ambient=ambient,
            control=control)
        if not constraints_satisfied:
            return None, False
        acc_metrics = self.metrics_accumulation.acc_metrics(aggregate=aggregate)
        
        expected_acc_metrics = acc_metrics.expected_value(probabilities=ambient_sample.normalized_weights)
        if self.constraint_accumulated is not None:
            if not self.constraint_accumulated.evaluate_satisfied(
                constraint_input=expected_acc_metrics):
                return None, False
        return expected_acc_metrics, True

class OptimizationMethod(Enum):
    GRID_SEARCH = "grid_search"
    SIMULTANEOUS_OPTIMIZATION = "simultaneous_optimization"
    LAGRANGIAN_RELAXATION = "lagrangian_relaxation"

class ControlPolicyOptimization(ABC):
    
    @abstractmethod
    def optimize_policy(self,
                        control_eval_system: ControlEvaluationSystem,
                        ambient_statistics: AmbientStatistics):
        pass

class ControlGridVectors:
    def __init__(self,
                 shape: Tuple[int, ...],
                 vectors: Sequence[np.ndarray]):
        if np.prod(shape) != len(vectors):
            raise ValueError("Input shape and vectors don't match.")
        self.shape = shape
        self.vectors = vectors

class GridSearch(ControlPolicyOptimization):
    def __init__(self,
                 max_num_amb_cond: int,
                 control_grid_vectors: Dict[Control, ControlGridVectors]):
        self.max_num_amb_cond = max_num_amb_cond
        self.control_grid_vectors = control_grid_vectors
    
    def optimize_policy(self,
                        ctrl_eval_sys: ControlEvaluationSystem,
                        ambient_statistics: AmbientStatistics):
        
        logger.info(f"GridSearch: Find optimal control policy")
        
        # Ambient conditions sample
        ambient_sample = ambient_statistics.systematic_sample(N_max=self.max_num_amb_cond)
        num_ambients = ambient_sample.N
        
        # Control grid-search
        ctrl_variables_order = list(self.control_grid_vectors.keys())
        # list of grid support vectors for each control input 
        ctrl_grid_vectors_list = [setpoints for ctrl_var in ctrl_variables_order for \
                                     setpoints in self.control_grid_vectors[ctrl_var].vectors]
        # Number of possible settings (values set for each control input)
        num_ctrl_settings = np.prod([len(ctrl_grid_vector) for ctrl_grid_vector in ctrl_grid_vectors_list])

        ctrl_shapes_dict={var: data.shape for var, data in self.control_grid_vectors.items()}

        # Logging
        logger.info(f"Number of ambient conditions: {num_ambients}.")
        logger.info(f"Number of ctrl settings: {num_ctrl_settings}.")
        num_eval = num_ctrl_settings * num_ambients
        
        if num_eval > MAX_NUM_EVALS:
            raise ValueError(f"The number of plant model evaluations {num_eval} is"
                             f" too large for grdis search optimization.")

        logger.info(f"Performing {num_eval} system evaluations.")
        
        # Ambient conditions to consider
        ambient = ambient_sample.ambient_support
        
        # Control setpoint combinations to consider
        control_settings_matrix = np.array(list(itertools.product(*ctrl_grid_vectors_list)))
        control = DataTable.from_matrix(data_matrix=control_settings_matrix,
                                        order=ctrl_variables_order,
                                        shapes_dict=ctrl_shapes_dict)
        # All scenarios (combinations of ambient conditions and control settings)
        ambient_tiled = ambient.repeat(num_ctrl_settings)
        control_repeated = control.tile(num_ambients)
        
        # Find admissible scenarios and their aggregates
        admissible_aggregates, instant_admissible = \
            ctrl_eval_sys.aggregate_from_amb_constr_eval(
            ambient=ambient_tiled,
            control=control_repeated)
        if len(instant_admissible) == 0:
            raise ValueError("Failed to find admissible policy due to instantaneous constraints-violation.")
        
        # Compute acc metrics for admissible aggregates
        admissible_metrics_accumulation = ctrl_eval_sys.metrics_accumulation.acc_metrics(
            aggregate=admissible_aggregates)
        pass
        
        # Admissible control settings for each ambient condition 
        ac_indices, ctrl_indices = np.unravel_index(instant_admissible,
                                                    shape=(num_ambients,
                                                           num_ctrl_settings))
        ambient_range = range(num_ambients)
        admissible_control_settings = [ [] for _ in ambient_range ]
        for ac_ind, ctrl_ind in zip(ac_indices, ctrl_indices):
            admissible_control_settings[ac_ind].append(ctrl_ind)  
        
        admissible_control_settings_shape = tuple(len(ctrl_settings) for \
                                                  ctrl_settings in admissible_control_settings)
        
        # Admissible ctrl policies based on admissible ctrl settings
        num_admissible_policies = np.prod(admissible_control_settings_shape)
        if num_admissible_policies > MAX_NUM_POLICIES:
            raise ValueError(f"The number of admissible policies {num_admissible_policies} is"
                             f" too large for grid search optimization.")

        logger.info(f"Evaluating {num_admissible_policies} control policies.")
        
        # Mapping between all scenarios and admissible scenarios
        num_admissible_scenarios = len(admissible_aggregates)
        admissible_range = range(num_admissible_scenarios)
        full_to_admissible_mapping = dict(zip(instant_admissible, admissible_range))

        ambient_tiled = np.tile(ambient_range, reps=num_admissible_policies)
        ctrl_settings_grid = np.meshgrid(*admissible_control_settings, indexing='ij')
        ctrl_settings_stacked = np.stack(ctrl_settings_grid, axis=-1).ravel()

        full_aggregate_indices = np.ravel_multi_index([ambient_tiled,
                                                       ctrl_settings_stacked],
                                                       dims=(num_ambients,
                                                             num_ctrl_settings))
        
        # Look-up table to replace full eval indices by admissible aggregate indices
        lut = np.empty(num_eval, dtype=int)
        for k, v in enumerate(instant_admissible):
            lut[v] = k
        # Sparse probabilities matrix
        rows = lut[full_aggregate_indices]
        cols = np.repeat(np.arange(num_admissible_policies), repeats=num_ambients)
        data = np.tile(ambient_sample.normalized_weights, reps=num_admissible_policies) 
        probabilities_coo = coo_matrix((data, (rows, cols)),
                                       shape=(num_admissible_scenarios, num_admissible_policies))
        
        # Expected metrics for each policy
        admissible_metrics_accumulations = admissible_metrics_accumulation.expected_value(
            probabilities=probabilities_coo
        )
        
        # Evaluate acc metrics constraint
        acc_metrics_admissible = np.arange(num_admissible_policies)
        if ctrl_eval_sys.constraint_accumulated is not None:
            acc_metrics_admissible = ctrl_eval_sys.constraint_accumulated.evaluate_satisfied(
                constraint_input=admissible_metrics_accumulations)
        if len(acc_metrics_admissible) == 0:
            raise ValueError("Failed to find feasible policy due to acc. constraints-violation.")
        
        admissible_multi_metrics_reduction = ctrl_eval_sys.multi_metrics_reduction.evaluate(
            acc_metrics=admissible_metrics_accumulations.extract(acc_metrics_admissible))
        
        # Find the optimal control strategy that satisfies the constraints
        if ctrl_eval_sys.multi_metrics_reduction.maximize:
            best_admissible_index = np.argmax(admissible_multi_metrics_reduction)
        else:
            best_admissible_index = np.argmax(admissible_multi_metrics_reduction)


        best_index = acc_metrics_admissible[best_admissible_index]

        # Specify discrete control strategy
        # best strategy among admissible strategies
        best_strategy_index = np.unravel_index(best_index, shape=admissible_control_settings_shape)
        # Retrieve control settings (flat index)
        best_strategy = [admissible_control_settings[ac][ind] for \
                         ac, ind in enumerate(best_strategy_index)]
        # list of indices for each control input
        ctrl_grid_vector_indices = np.unravel_index(indices=best_strategy,
                                       shape=(tuple(len(vec) for vec in ctrl_grid_vectors_list)))
        # Transform to index-matrix: rows -> ambient condition indices, columns -> control input indices
        ctrl_grid_vector_ind_matrix = np.array(ctrl_grid_vector_indices).T
        # Retrieve control input values from grid vectors
        control_matrix = [[ctrl_grid_vector[ctrl_grid_vector_ind_matrix[amb, ctrl]] for \
                          ctrl, ctrl_grid_vector in enumerate(ctrl_grid_vectors_list)] for \
                            amb in ambient_range]
        # Create data table
        control_out_data = DataTable.from_matrix(data_matrix=np.array(control_matrix),
                                                 order=ctrl_variables_order,
                                                 shapes_dict=ctrl_shapes_dict)        
            
        # Build control policy
        optimal_policy = DiscreteControlPolicy(
            name="optimized_discrete_control_policy",
            ambient_support_data=ambient,
            control_out_data=control_out_data)
    
        final_acc_metrics = ctrl_eval_sys.expected_acc_metrics(
            ambient_statistics=ambient_statistics,
            control_policy=optimal_policy)
        logger.info(f"Grid-search optimization final control policy:\n"
                    f"{optimal_policy}"
                    f"Final accumulated metrics:\n"
                    f"{final_acc_metrics}")

        return optimal_policy


def grid_search_from_dict(param_dict: Dict[str, Dict | Any]):
    max_num_amb_cond = param_dict["max_num_ambient_conditions"]
    control_grid_vectors = {}
    for ctrl_var, setpoint_vectors in param_dict["control_grid_vectors"].items():
        control_grid_vectors[Control(ctrl_var)] = ControlGridVectors(
            shape=tuple(setpoint_vectors["shape"]),
            vectors=setpoint_vectors["data"])
    return GridSearch(
        max_num_amb_cond=max_num_amb_cond,
        control_grid_vectors=control_grid_vectors)

class ContinuousOptimizationManager:
    def __init__(self,
                 control_eval_system: ControlEvaluationSystem,
                 ambient_statistics: AmbientStatistics,
                 max_num_amb_cond: int | None = None):
        self._control_eval_system = control_eval_system
        self._ambient_sample = ambient_statistics.systematic_sample(
            N_max=max_num_amb_cond)
        self._control_policy = default_discrete_policy(
            control_shapes=self._control_eval_system.plant_model.input_interface.shapes[Control],
            ambient_support=self._ambient_sample.ambient_support
        )
        self._control_order = self._control_policy.control_out_data().order
        self._control_shapes = self._control_policy.control_out_data().shapes()
        self._control_len = len(self._control_policy.control_out_data())


    # TODO: Make sure that caching works.
    @lru_cache(maxsize=None)
    def single_aggregate_w_cache(self,
                                 i_ac: int,
                                 ctrl_as_tuple: Tuple[float, ...]):
        control_setpoints = DataTable.from_vector(
            data_vector=np.array(ctrl_as_tuple),
            order=self._control_order,
            shapes_dict=self._control_shapes)

        return self._control_eval_system.aggregate_from_amb(
            ambient=self._ambient_sample.ambient_support.get_point(i_ac),
            control=control_setpoints)
        
    @lru_cache(maxsize=None)
    def single_acc_metric_w_cache(self,
                                  i_ac: int,
                                  ctrl_as_tuple: Tuple[float, ...]):
        aggregate = self.single_aggregate_w_cache(
            i_ac=i_ac,
            ctrl_as_tuple=ctrl_as_tuple)
        
        return self._control_eval_system.metrics_accumulation.acc_metrics(
            aggregate=aggregate)

    @lru_cache(maxsize=None)
    def all_aggregates_w_cache(self,
                           ctrl_as_tuple: Tuple[float, ...]):
        control_setpoints_data = DataTable.from_vector(
            data_vector=np.array(ctrl_as_tuple),
            order=self._control_order,
            shapes_dict=self._control_shapes,
            num_points=self._control_len)

        aggregate_list = []
        for i_ac, ambient in enumerate(self._ambient_sample.ambient_support):
            control_setpoints = control_setpoints_data.get_point(i_ac)
            aggregate = self._control_eval_system.aggregate_from_amb(
                ambient=ambient,
                control=control_setpoints)
            aggregate_list.append(aggregate)
        return DataTable.from_data_points(aggregate_list)

    @lru_cache(maxsize=None)
    def expected_acc_metrics_w_cache(self,
                                     ctrl_as_tuple: Tuple[float, ...]):
        aggregate_support = self.all_aggregates_w_cache(ctrl_as_tuple=ctrl_as_tuple)
        discrete_aggregate_statistics = DiscreteAmbientStatistics(
            name="",
            ambient_support=aggregate_support,
            probabilities=self._ambient_sample.normalized_weights)        

        expected_acc_metrics = self._control_eval_system.metrics_accumulation.expected_acc_metrics(
            aggregate_statistics=discrete_aggregate_statistics)
        
        return expected_acc_metrics

class SimultaneousOptimization(ControlPolicyOptimization):
    def __init__(self,
                 max_num_amb_cond: int,
                 scipy_method: str,
                 scipy_options: Dict[str, Any]):
        self.max_num_amb_cond = max_num_amb_cond
        self.scipy_method = scipy_method
        self.scipy_options = scipy_options

    def optimize_policy(self,
                        control_eval_system: ControlEvaluationSystem,
                        ambient_statistics: AmbientStatistics):
        
        # OptimizationManager
        opt_mgr = ContinuousOptimizationManager(
            control_eval_system=control_eval_system,
            ambient_statistics=ambient_statistics,
            max_num_amb_cond=self.max_num_amb_cond)
        
        # Optimization functions with cache
        all_aggregates_w_cache = lambda x : opt_mgr.all_aggregates_w_cache(tuple(x))        
        expected_acc_metrics_w_cache = lambda x : opt_mgr.expected_acc_metrics_w_cache(tuple(x))        
        
        # COST FUNCTION
        # based on accumulated metrics
        cost_function = \
            opt_mgr._control_eval_system.multi_metrics_reduction.cost_function(
                eval_acc_metrics_from_x=expected_acc_metrics_w_cache)

        # CONSTRAINTS
        constraints = []
        bounds = None
        # Order and shapes of control variables
        x_order = opt_mgr._control_order
        x_shapes_dict=opt_mgr._control_shapes

        # CONTROL CONSTRAINT
        if opt_mgr._control_eval_system.constraint_control is not None:
            num_ac = opt_mgr._control_len
            control_constraint_object = \
                opt_mgr._control_eval_system.constraint_control.scipy_object(
                x_order=x_order,
                x_shapes_dict=x_shapes_dict,
                num_points=num_ac,
            )
            if isinstance(control_constraint_object, Bounds):
                bounds = control_constraint_object
            else:
                constraints.append(control_constraint_object)

        # AGGREGATE CONSTRAINT
        if opt_mgr._control_eval_system.constraint_aggregate is not None:
            aggregate_constraint = \
                opt_mgr._control_eval_system.constraint_aggregate.scipy_object(
                x_order=x_order,
                x_shapes_dict=x_shapes_dict,
                num_points=num_ac,
                x_constraint_evaluation=all_aggregates_w_cache
            )
            constraints.append(aggregate_constraint)

        # ACCUMULATED CONSTRAINT
        if opt_mgr._control_eval_system.constraint_accumulated is not None:
            acc_metrics_constraint = \
                opt_mgr._control_eval_system.constraint_accumulated.scipy_object(
                    x_order=x_order,
                    x_shapes_dict=x_shapes_dict,                     
                    x_constraint_evaluation=expected_acc_metrics_w_cache)
            constraints.append(acc_metrics_constraint)
        
        # SCIPY-OPTIMIZATION
        x0 = opt_mgr._control_policy.control_out_data().to_vector()
        res = minimize(cost_function,
                       x0,
                       method=self.scipy_method,
                       bounds=bounds,
                       constraints=constraints,
                       options=self.scipy_options)
        # TODO: Fix retrieval of DataTable properties:
        optimal_control_setpoints = DataTable.from_vector(
            data_vector=res.x,
            order=opt_mgr._control_order,
            shapes_dict=opt_mgr._control_shapes,
            num_points=opt_mgr._control_len)
        optimal_control_policy = DiscreteControlPolicy(
            name="optimal_policy",
            ambient_support_data=opt_mgr._ambient_sample.ambient_support,
            control_out_data=optimal_control_setpoints            
        )
        
        final_acc_metrics = opt_mgr._control_eval_system.expected_acc_metrics(
                        ambient_statistics=ambient_statistics,
                        control_policy=optimal_control_policy)
        logger.info(f"Simultaneous optimization final control policy:\n"
                    f"{optimal_control_policy}"
                    f"Final accumulated metrics:\n"
                    f"{final_acc_metrics}")
        return optimal_control_policy
     
def simultaneous_optimization_from_dict(param_dict: Dict[str, Any]):
    max_num_amb_cond = param_dict["max_num_ambients"]
    scipy_method = param_dict["scipy_method"]
    scipy_options = param_dict["scipy_options"]
    return SimultaneousOptimization(
        max_num_amb_cond=max_num_amb_cond,
        scipy_method=scipy_method,
        scipy_options=scipy_options)

class LagrangianRelaxation:
    def __init__(self,
                 max_num_amb_cond: int,
                 alpha_0: float,
                 max_iter: int,
                 constraint_tol: float,
                 lagrangian_lambda_tol: float,
                 scipy_method: str,
                 scipy_options: Dict[str, Any]):
        self.max_num_amb_cond = max_num_amb_cond
        self.alpha_0 = alpha_0
        self.max_iter = max_iter
        self.constraint_tol = constraint_tol
        self.lagrangian_lambda_tol = lagrangian_lambda_tol
        self.scipy_method = scipy_method
        self.scipy_options = scipy_options

class LagrangianRelaxation(ControlPolicyOptimization):
    def __init__(self,
                 max_num_amb_cond: int,
                 alpha_0: float,
                 max_iter: int,
                 constraint_tol: float,
                 lagrangian_lambda_tol: float,
                 scipy_method: str,
                 scipy_options: Dict[str, Any]):
        self._max_num_amb_cond = max_num_amb_cond
        self._alpha_0 = alpha_0
        self._max_iter = max_iter
        self._constraint_tol = constraint_tol
        self._lagrangian_lambda_tol = lagrangian_lambda_tol
        self._scipy_method = scipy_method
        self._scipy_options = scipy_options
        
    def optimize_policy(self,
                        control_eval_system: ControlEvaluationSystem,
                        ambient_statistics: AmbientStatistics):
        
        # OptimizationManager
        opt_mgr = ContinuousOptimizationManager(
            control_eval_system=control_eval_system,
            ambient_statistics=ambient_statistics,
            max_num_amb_cond=self._max_num_amb_cond)
        
        control_setpoints = opt_mgr._control_policy.control_out_data()
                
        ambient_sample = ambient_statistics.systematic_sample(N_max=self._max_num_amb_cond)

        if control_eval_system.constraint_accumulated is not None:
            upper_bound_constraints = control_eval_system.constraint_accumulated.upper_bound_constraints()
            if len(upper_bound_constraints) > 1:
                # TODO: Check this
                raise NotImplementedError("Currently only one set of accumulated metrics bounds allowed.")
            ub_constraint = upper_bound_constraints[0]
            lagrangian_lambdas = np.zeros_like(ub_constraint.upper_bound, dtype=float)
            lagrangian_lambdas_low = lagrangian_lambdas.copy()
            lagrangian_lambdas_high = np.full_like(lagrangian_lambdas, fill_value=np.inf)
            
        # outer-iteration counter
        for t in range(self._max_iter):
            if control_eval_system.constraint_accumulated is not None:
                expected_constr_eval = np.zeros_like(lagrangian_lambdas)
            for i_ac, (prob_weight, ambient) in enumerate(ambient_sample.weighted_variables_iter()):

                # Optimization functions with cache
                aggregate_w_cache = lambda x, i_ac=i_ac : \
                    opt_mgr.single_aggregate_w_cache(i_ac, tuple(x))        
                acc_metrics_w_cache = lambda x, i_ac=i_ac : \
                    opt_mgr.single_acc_metric_w_cache(i_ac, tuple(x))  
                # Separate optimization for each ambient condition
                # COST FUNCTION
                def cost_function(ctrl_setpoints_vec, i_ac=i_ac):
                    acc_metrics = acc_metrics_w_cache(ctrl_setpoints_vec)
                    scalar_objective = opt_mgr._control_eval_system.multi_metrics_reduction.evaluate(
                        acc_metrics=acc_metrics)
                    # First assume that we have an objective function (which we want to maximise),
                    # and upper-bound constraints which must not be exceded.
                    if not opt_mgr._control_eval_system.multi_metrics_reduction.maximize:
                        scalar_objective *= -1   
                    if control_eval_system.constraint_accumulated is not None:
                        # Lagrangian relaxation
                        scalar_objective -= lagrangian_lambdas.dot(ub_constraint.constraint_fun(acc_metrics))
                    # Scipy will minimize a cost function, hence take the negative value
                    return - scalar_objective
                
                # INSTANTANEOUS CONSTRAINTS
                instant_constraints = []
                bounds = None
                # Order and shapes of control variables
                x_order = opt_mgr._control_order
                x_shapes_dict=opt_mgr._control_shapes
                
                # CONTROL CONSTRAINTS
                if opt_mgr._control_eval_system.constraint_control is not None:
                    control_constraint_object = opt_mgr._control_eval_system.constraint_control.scipy_object(
                        x_order=x_order,
                        x_shapes_dict=x_shapes_dict
                    )
                    if isinstance(control_constraint_object, Bounds):
                        bounds = control_constraint_object
                    else:
                        instant_constraints.append(control_constraint_object)

                # AGGREGATE CONSTRAINT
                if opt_mgr._control_eval_system.constraint_aggregate is not None:
                    aggregate_constraint = opt_mgr._control_eval_system.constraint_aggregate.scipy_object(
                        x_order=x_order,
                        x_shapes_dict=x_shapes_dict,
                        x_constraint_evaluation=aggregate_w_cache
                    )
                    instant_constraints.append(aggregate_constraint)
                # Minimize Lagrangian
                # Initial value
                x0 = control_setpoints.get_point(i_ac).to_vector()
                res = minimize(cost_function,
                               x0,
                               method=self._scipy_method,
                               bounds=bounds,
                               constraints=instant_constraints,
                               options=self._scipy_options)
                control_setpoints.update_point_from_vector(ind=i_ac,
                                                           vector=res.x)
                
                # Recompute final accumulated metrics
                acc_metrics = opt_mgr._control_eval_system.acc_metrics_from_amb(
                    ambient=ambient,
                    control=control_setpoints.get_point(i_ac))
                if control_eval_system.constraint_accumulated is None:
                    # We are done
                    break
                # Constraint evaluation
                expected_constr_eval += prob_weight * ub_constraint.constraint_fun(acc_metrics)
            logger.debug(f"Iter {t}, before Lagrangian update: lambda = {lagrangian_lambdas},  constraint_eval = {expected_constr_eval}")
            constr_violation = expected_constr_eval - ub_constraint.upper_bound
            lagrangian_lambdas_low = np.where(constr_violation > 0, lagrangian_lambdas, lagrangian_lambdas_low)
            lagrangian_lambdas_high = np.where(constr_violation < 0, lagrangian_lambdas, lagrangian_lambdas_high)
            
            for i_constr, (lam_low, lam_high) in enumerate(zip(lagrangian_lambdas_low, lagrangian_lambdas_high)):
                if lam_high < np.inf:
                    lagrangian_lambdas[i_constr] = (lam_high + lam_low) / 2
                else:
                    step = np.sign(constr_violation[i_constr])
                    lagrangian_lambdas[i_constr] += step
            if np.all(constr_violation < self._constraint_tol) and \
                np.all(np.abs(lagrangian_lambdas_high - lagrangian_lambdas_low) < self._lagrangian_lambda_tol):
                # Converged to stable lagrangian lamdas
                break

        final_acc_metrics = opt_mgr._control_eval_system.expected_acc_metrics(
                        ambient_statistics=ambient_statistics,
                        control_policy=opt_mgr._control_policy)
        logger.info(f"Lagrangian relaxation optimization finished after {t + 1} external iterations.\n"
                    f"Final control policy:\n"
                    f"{opt_mgr._control_policy}"
                    f"Final accumulated metrics:\n"
                    f"{final_acc_metrics}")
        return opt_mgr._control_policy
        
def lagrangian_relaxation_from_dict(param_dict: Dict[str, Any]):
    max_num_amb_cond = param_dict["max_num_ambients"]
    alpha_0 = param_dict["alpha_0"]
    max_iter = param_dict["max_iter"]
    constraint_tol = param_dict["constraint_tol"]
    lagrangian_lambda_tol = param_dict["lagrangian_lambda_tol"]
    scipy_method = param_dict["scipy_method"]
    scipy_options = param_dict["scipy_options"]
    return LagrangianRelaxation(
        max_num_amb_cond=max_num_amb_cond,
        alpha_0=alpha_0,
        max_iter=max_iter,
        constraint_tol=constraint_tol,
        lagrangian_lambda_tol=lagrangian_lambda_tol,
        scipy_method=scipy_method,
        scipy_options=scipy_options)

