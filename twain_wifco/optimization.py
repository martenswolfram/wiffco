from typing import Dict, Any, Tuple, Sequence
from abc import ABC, abstractmethod
import numpy as np
import logging
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from functools import lru_cache
import itertools
from enum import Enum
from twain_wifco.interface import (
    Ambient,
    Control,
    get_abs_tol,
    AccumulatedMetric,
    DataTable
)
from twain_wifco.statistics import (
    AmbientStatistics,
    DiscreteAmbientStatistics
)
from twain_wifco.control_policy import (
    DiscreteControlPolicy,
    default_discrete_policy
)
from twain_wifco.plant_model import PlantModel
from twain_wifco.aggregation import Aggregation
from twain_wifco.metrics_accumulation import MetricsAccumulation
from twain_wifco.constraint import (
    ControlConstraint,
    AggregateConstraint,
    AccumulatedConstraint,
    TwoSidedBound
)
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
        admissible_aggregates = control_admissible_aggregates.extract(
            indices=aggr_admissible_among_ctrl_admissible)
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
                        control_eval_system: ControlEvaluationSystem,
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
            control_eval_system.aggregate_from_amb_constr_eval(
            ambient=ambient_tiled,
            control=control_repeated)
        if len(instant_admissible) == 0:
            raise ValueError("Failed to find admissible policy due to instantaneous constraints-violation.")
        
        # Compute acc metrics for admissible aggregates
        admissible_metrics_accumulation = control_eval_system.metrics_accumulation.acc_metrics(
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
        if control_eval_system.constraint_accumulated is not None:
            acc_metrics_admissible = control_eval_system.constraint_accumulated.evaluate_satisfied(
                constraint_input=admissible_metrics_accumulations)
        if len(acc_metrics_admissible) == 0:
            raise ValueError("Failed to find feasible policy due to acc. constraints-violation.")
        
        admissible_multi_metrics_reduction = control_eval_system.multi_metrics_reduction.evaluate(
            acc_metrics=admissible_metrics_accumulations.extract(acc_metrics_admissible))
        
        # Find the optimal control strategy that satisfies the constraints
        if control_eval_system.multi_metrics_reduction.maximize:
            best_admissible_index = np.argmax(admissible_multi_metrics_reduction)
        else:
            best_admissible_index = np.argmin(admissible_multi_metrics_reduction)


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
    
        final_acc_metrics = control_eval_system.expected_acc_metrics(
            ambient_statistics=ambient_statistics,
            control_policy=optimal_policy)
        logger.info(f"Grid-search optimization final control policy:\n"
                    f"{optimal_policy}"
                    f"Final accumulated metrics:\n"
                    f"{final_acc_metrics}")

        return optimal_policy


def grid_search_from_dict(param_dict: Dict[str, Dict | Any]):
    max_num_amb_cond = param_dict["max_num_ambients"]
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
                 control_shapes: Dict[Control, Tuple[int, ...]],
                 ambient_statistics: AmbientStatistics,
                 max_num_amb_cond: int | None = None):
        self._control_eval_system = control_eval_system
        self._ambient_sample = ambient_statistics.systematic_sample(
            N_max=max_num_amb_cond)
        
        self.num_ambient = self._ambient_sample.N
        self._control_shapes = control_shapes
        
        # Default evaluation to specify data formats
        self._control_policy = default_discrete_policy(
            control_shapes=self._control_shapes,
            ambient_support=self._ambient_sample.ambient_support
        )
        self._control_order = self._control_policy.control_out_data().order
        
        # default aggregate
        ambient = self._ambient_sample.ambient_support
        control = self._control_policy.get_control(ambient=ambient)
        self._control_order = control.order
        self._control_shapes = control.shapes()

        model_output = self._control_eval_system.plant_model.evaluate(
            meteorological=ambient,
            control=control
        )
        aggregate = self._control_eval_system.aggregation.compute_aggregate(
            ambient=ambient,
            control=control,
            model_output=model_output
        )
        self._aggregate_order = aggregate.order
        self._aggregate_shapes = aggregate.shapes()

        # default metrics accumulation
        acc_metrics = self._control_eval_system.metrics_accumulation.acc_metrics(
            aggregate=aggregate
        )
        self._acc_metrics_order = acc_metrics.order
        self._acc_metrics_shapes = acc_metrics.shapes()


    # TODO: Make sure that caching works.
    @lru_cache(maxsize=None)
    def single_aggregate_w_cache(self,
                                 i_ac: int,
                                 ctrl_as_tuple: Tuple[float, ...]):
        ambient = self._ambient_sample.ambient_support.extract(indices=i_ac)
        control = DataTable.from_vector(
            data_vector=np.array(ctrl_as_tuple),
            order=self._control_order,
            shapes_dict=self._control_shapes,
            num_points=1)
        model_output = self._control_eval_system.plant_model.evaluate(
             meteorological=ambient,
             control=control)
        return self._control_eval_system.aggregation.compute_aggregate(
            model_output=model_output,
            ambient=ambient,
            control=control)
        
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
        control = DataTable.from_vector(
            data_vector=np.array(ctrl_as_tuple),
            order=self._control_order,
            shapes_dict=self._control_shapes,
            num_points=self.num_ambient)

        model_output = self._control_eval_system.plant_model.evaluate(
            meteorological=self._ambient_sample.ambient_support,
            control=control
        )
        return self._control_eval_system.aggregation.compute_aggregate(
            ambient=self._ambient_sample.ambient_support,
            control=control,
            model_output=model_output
        )

    @lru_cache(maxsize=None)
    def expected_acc_metrics_w_cache(self,
                                     ctrl_as_tuple: Tuple[float, ...]):
        aggregate = self.all_aggregates_w_cache(ctrl_as_tuple=ctrl_as_tuple)
        acc_metrics = self._control_eval_system.metrics_accumulation.acc_metrics(
            aggregate=aggregate
        )
        expected_acc_metrics = acc_metrics.expected_value(
            probabilities=self._ambient_sample.normalized_weights[:, np.newaxis])        
        return expected_acc_metrics

class SimultaneousOptimization(ControlPolicyOptimization):
    def __init__(self,
                 control_shapes: Dict[Control, Tuple[int, ...]], 
                 scipy_method: str,
                 scipy_options: Dict[str, Any],
                 max_num_amb_cond: int | None = None):
        self._control_shapes = control_shapes
        self._scipy_method = scipy_method
        self._scipy_options = scipy_options
        self._max_num_amb_cond = max_num_amb_cond
        
    def optimize_policy(self,
                        control_eval_system: ControlEvaluationSystem,
                        ambient_statistics: AmbientStatistics):
        
        # OptimizationManager
        opt_mgr = ContinuousOptimizationManager(
            control_eval_system=control_eval_system,
            control_shapes=self._control_shapes,
            ambient_statistics=ambient_statistics,
            max_num_amb_cond=self._max_num_amb_cond)
        
        # Optimization functions with cache
        all_aggregates_w_cache = lambda x : opt_mgr.all_aggregates_w_cache(tuple(x))        
        expected_acc_metrics_w_cache = lambda x : opt_mgr.expected_acc_metrics_w_cache(tuple(x))        
        
        # COST FUNCTION
        # based on accumulated metrics
        cost_function = \
            opt_mgr._control_eval_system.multi_metrics_reduction.cost_function(
                eval_metrics_from_x=expected_acc_metrics_w_cache)

        # CONTROL CONSTRAINT
        if opt_mgr._control_eval_system.constraint_control is not None:
            bounds = opt_mgr._control_eval_system.constraint_control.scipy_bounds(
                control_order=opt_mgr._control_order,
                control_shapes_dict=opt_mgr._control_shapes,
                num_points=opt_mgr.num_ambient)
        else:
            bounds = None
        
        scipy_constraints = []
        
        # AGGREGATE CONSTRAINT
        if opt_mgr._control_eval_system.constraint_aggregate is not None:
            scipy_constraints.append(
                opt_mgr._control_eval_system.constraint_aggregate.scipy_constraint(
                    aggregate_shapes_dict=opt_mgr._aggregate_shapes,
                    aggregate_evaluation=all_aggregates_w_cache,
                    num_points=opt_mgr.num_ambient)
            )

        # ACCUMULATED CONSTRAINT
        if opt_mgr._control_eval_system.constraint_accumulated is not None:
            scipy_constraints.append(
                opt_mgr._control_eval_system.constraint_accumulated.scipy_constraint(
                    accumulated_order=opt_mgr._acc_metrics_order,
                    accumulated_shapes_dict=opt_mgr._acc_metrics_shapes,
                    accumulated_evaluation=expected_acc_metrics_w_cache)
            )
            
        # SCIPY-OPTIMIZATION
        x0 = opt_mgr._control_policy.control_out_data().to_vector()
        res = minimize(cost_function,
                       x0,
                       method=self._scipy_method,
                       bounds=bounds,
                       constraints=scipy_constraints,
                       options=self._scipy_options)
        
        # Create optimal control policy
        optimal_control = DataTable.from_vector(
            data_vector=res.x,
            order=opt_mgr._control_order,
            shapes_dict=opt_mgr._control_shapes,
            num_points=opt_mgr.num_ambient)
        
        optimal_control_policy = DiscreteControlPolicy(
            name="optimal_policy",
            ambient_support_data=opt_mgr._ambient_sample.ambient_support,
            control_out_data=optimal_control            
        )
        
        final_acc_metrics = opt_mgr._control_eval_system.expected_acc_metrics(
                        ambient_statistics=ambient_statistics,
                        control_policy=optimal_control_policy)
        logger.info(f"Simultaneous optimization final control policy:\n"
                    f"{optimal_control_policy}"
                    f"Final accumulated metrics:\n"
                    f"{final_acc_metrics}")
        return optimal_control_policy
     
def simultaneous_optimization_from_dict(param_dict: Dict[str, Any | Dict]):
    max_num_amb_cond = param_dict.get("max_num_ambients")
    control_shapes = {Control(ctrl_var): tuple(shape) for \
                      ctrl_var, shape in param_dict["control_shapes"].items()}
    scipy_method = param_dict["scipy_method"]
    scipy_options = param_dict["scipy_options"]
    return SimultaneousOptimization(
        control_shapes=control_shapes,
        scipy_method=scipy_method,
        scipy_options=scipy_options,
        max_num_amb_cond=max_num_amb_cond)

class LagrangianLambda:
    def __init__(self,
                 bounds: Dict[AccumulatedMetric, TwoSidedBound],
                 acc_metrics_shapes: Dict[AccumulatedMetric, Tuple[int, ...]],
                 lambda_abs_tol: float):
        
        # Shapes of constrained accumulated metrics
        upper_bound_shapes = {}
        lower_bound_shapes = {}
        for acc_metric, bound in bounds.items():
            if not np.isinf(bound.upper):
                upper_bound_shapes[acc_metric] = acc_metrics_shapes[acc_metric]
            if not np.isneginf(bound.lower):
                lower_bound_shapes[acc_metric] = acc_metrics_shapes[acc_metric]
            
        # Order of accumulated metrics for vectorization
        self.upper_bound_order = list(upper_bound_shapes.keys())
        self.lower_bound_order = list(lower_bound_shapes.keys())
        
        # Flat sizes for accumulated metrics
        upper_sizes = \
            list(np.prod(shape) for \
                 shape in upper_bound_shapes.values())
        lower_sizes = \
            list(np.prod(shape) for \
                 shape in lower_bound_shapes.values())
        upper_size = np.sum(upper_sizes, dtype=int)
        lower_size = np.sum(lower_sizes, dtype=int)
        
        # Vectorize bounds (combine upper and lower bounds)
        upper_bound_vector = np.repeat(
            np.array([bounds[var].upper for var in self.upper_bound_order]),
            repeats=upper_sizes)
        lower_bound_vector = np.repeat(
            np.array([bounds[var].lower for var in self.lower_bound_order]),
            repeats=lower_sizes)
        # Combine as upper bound vector by inverting sign of lower bound 
        self.upper_bound_vector = np.hstack([upper_bound_vector, - lower_bound_vector])
        
        # Tolerances for vectors (combine upper and lower bounds)
        upper_constraint_tol_vector = np.repeat(
            np.array([get_abs_tol(var) for var in self.upper_bound_order]),
            repeats=upper_sizes)
        lower_constraint_tol_vector = np.repeat(
            np.array([get_abs_tol(var) for var in self.lower_bound_order]),
            repeats=lower_sizes)
        self.constraint_tol_vector = np.hstack([upper_constraint_tol_vector, lower_constraint_tol_vector])
        
        # Lagrangian lambdas, combined for upper and lower bounds
        self._lagrangian_lambda = np.zeros(shape=(upper_size + lower_size,), dtype=float)
        self._lambda_low = np.zeros_like(self._lagrangian_lambda)
        self._lambda_high = np.full_like(self._lagrangian_lambda, fill_value=np.inf)
        self._lambda_abs_tol = lambda_abs_tol

    def vector_eval(self, acc_metric: DataTable[AccumulatedMetric]):
        # Invert sign for lower bounds 
        return np.hstack([acc_metric.to_vector(order=self.upper_bound_order),
                          - acc_metric.to_vector(order=self.lower_bound_order)])

    def dot_product(self, acc_metric: DataTable[AccumulatedMetric]):
        return self._lagrangian_lambda.dot(
            self.vector_eval(acc_metric=acc_metric))
        
    def process_constraint_violation(self, acc_metric: DataTable[AccumulatedMetric]):
        # Compute violation (upper and lower bounds combined)
        violation = self.vector_eval(acc_metric=acc_metric) - self.upper_bound_vector
        
        # Update clipping
        self._lambda_low = np.where(violation > 0,
                                         self._lagrangian_lambda,
                                         self._lambda_low)
        self._lambda_high = np.where(violation < 0,
                                          self._lagrangian_lambda,
                                          self._lambda_high)
        # Bisection approach
        for i_constr, (lam_low, lam_high) in enumerate(zip(self._lambda_low,
                                                           self._lambda_high)):
            if lam_high < np.inf:
                self._lagrangian_lambda[i_constr] = (lam_high + lam_low) / 2
            else:
                step = np.sign(violation[i_constr])
                self._lagrangian_lambda[i_constr] += step * 0.01
        if np.all(violation < self.constraint_tol_vector) and \
            np.all(np.abs(self._lambda_high - self._lambda_low) < \
                   self._lambda_abs_tol):
            # Convergence
            return True
        else:
            # No convergence
            return False

class LagrangianRelaxation(ControlPolicyOptimization):
    def __init__(self,
                 control_shapes: Dict[Control, Tuple[int, ...]], 
                 alpha_0: float,
                 max_iter: int,
                 constraint_tol: float,
                 lagrangian_lambda_tol: float,
                 scipy_method: str,
                 scipy_options: Dict[str, Any],
                 max_num_amb_cond: int | None = None):
        self._control_shapes = control_shapes
        self._alpha_0 = alpha_0
        self._max_iter = max_iter
        self._constraint_tol = constraint_tol
        self._lagrangian_lambda_tol = lagrangian_lambda_tol
        self._scipy_method = scipy_method
        self._scipy_options = scipy_options
        self._max_num_amb_cond = max_num_amb_cond
        
    def optimize_policy(self,
                        control_eval_system: ControlEvaluationSystem,
                        ambient_statistics: AmbientStatistics):
        
        # OptimizationManager
        opt_mgr = ContinuousOptimizationManager(
            control_eval_system=control_eval_system,
            control_shapes=self._control_shapes,
            ambient_statistics=ambient_statistics,
            max_num_amb_cond=self._max_num_amb_cond)
                
        control = opt_mgr._control_policy.control_out_data()
        acc_constraint = control_eval_system.constraint_accumulated is not None
        if acc_constraint:
            acc_bounds = control_eval_system.constraint_accumulated.bounds()
            llambda = LagrangianLambda(
                bounds=acc_bounds,
                acc_metrics_shapes=opt_mgr._acc_metrics_shapes,
                lambda_abs_tol=self._lagrangian_lambda_tol
            )
        
        # caching functions for each ambient condition
        aggregate_w_cache = []
        acc_metric_w_cache = []
        for i_ac in range(opt_mgr.num_ambient):
            aggregate_w_cache.append(
                lambda x, i_ac=i_ac : opt_mgr.single_aggregate_w_cache(i_ac, tuple(x)))        
            acc_metric_w_cache.append(
                lambda x, i_ac=i_ac : opt_mgr.single_acc_metric_w_cache(i_ac, tuple(x)))

        # outer-iteration counter
        for t in range(self._max_iter):
            for i_ac in range(opt_mgr.num_ambient):
                # Separate optimization for each ambient condition
                # COST FUNCTION
                def cost_function(ctrl_setpoints_vec, i_ac=i_ac):
                    acc_metric = acc_metric_w_cache[i_ac](ctrl_setpoints_vec)
                    scalar_objective = \
                        opt_mgr._control_eval_system.multi_metrics_reduction.evaluate(
                        acc_metrics=acc_metric)
                    # First assume that we have an objective function (i.e. which we want to maximise),
                    # and upper-bound constraints which must not be exceded.
                    if not opt_mgr._control_eval_system.multi_metrics_reduction.maximize:
                        scalar_objective *= -1   
                    if acc_constraint:
                        # Lagrangian relaxation
                        scalar_objective -= llambda.dot_product(acc_metric) 
                    # Scipy will minimize a cost function, hence take the negative value
                    return - scalar_objective
                
                # INSTANTANEOUS CONSTRAINTS
                # CONTROL CONSTRAINTS
                if opt_mgr._control_eval_system.constraint_control is not None:
                    bounds = opt_mgr._control_eval_system.constraint_control.scipy_bounds(
                        control_order=opt_mgr._control_order,
                        control_shapes_dict=opt_mgr._control_shapes,
                        num_points=1
                    )
                else:
                    bounds = None

                # AGGREGATE CONSTRAINT
                if opt_mgr._control_eval_system.constraint_aggregate is not None:
                    aggregate_constraint = opt_mgr._control_eval_system.constraint_aggregate.scipy_constraint(
                        aggregate_shapes_dict=opt_mgr._aggregate_shapes,
                        aggregate_evaluation=aggregate_w_cache[i_ac],
                        num_points=1
                    )
                else:
                    aggregate_constraint = None
                
                # Minimize Lagrangian
                # Initial value
                x0 = control.to_vector(point_index=i_ac)
                res = minimize(cost_function,
                               x0,
                               method=self._scipy_method,
                               bounds=bounds,
                               constraints=aggregate_constraint,
                               options=self._scipy_options)
                control.update_point_from_vector(
                    order=opt_mgr._control_order,
                    point_index=i_ac,
                    vector=res.x)
                                    
            if acc_constraint:
                # Compute final expected accumulated metrics for optimized control (without lagrangian multipliers)
                expected_acc_metric = opt_mgr._control_eval_system.expected_acc_metrics(
                    ambient_statistics=ambient_statistics,
                    control_policy=opt_mgr._control_policy)
                # Evaluate constraint violation and update lagrangian lambdas
                if llambda.process_constraint_violation(acc_metric=expected_acc_metric):
                    # Converged
                    break
                
            else:
                # We are done
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
    max_num_amb_cond = param_dict.get("max_num_ambients")
    control_shapes = {Control(ctrl_var): tuple(shape) for \
                      ctrl_var, shape in param_dict["control_shapes"].items()}
    alpha_0 = param_dict["alpha_0"]
    max_iter = param_dict["max_iter"]
    constraint_tol = param_dict["constraint_tol"]
    lagrangian_lambda_tol = param_dict["lagrangian_lambda_tol"]
    scipy_method = param_dict["scipy_method"]
    scipy_options = param_dict["scipy_options"]
    return LagrangianRelaxation(
        control_shapes=control_shapes,
        alpha_0=alpha_0,
        max_iter=max_iter,
        constraint_tol=constraint_tol,
        lagrangian_lambda_tol=lagrangian_lambda_tol,
        scipy_method=scipy_method,
        scipy_options=scipy_options,
        max_num_amb_cond=max_num_amb_cond)

def control_optimization_from_dict(
    param_dict: Dict[str, Any]):
    optimization_method = OptimizationMethod(param_dict["optimization_method"])
    if optimization_method == OptimizationMethod.GRID_SEARCH:
        return grid_search_from_dict(param_dict=param_dict)
    elif optimization_method == OptimizationMethod.SIMULTANEOUS_OPTIMIZATION:
        return simultaneous_optimization_from_dict(param_dict=param_dict)
    elif optimization_method == OptimizationMethod.LAGRANGIAN_RELAXATION:
        return lagrangian_relaxation_from_dict(param_dict=param_dict)
    else:
        raise NotImplementedError("Only grid-search, simultaneous opt. and lagrangian relaxation implemented.")
