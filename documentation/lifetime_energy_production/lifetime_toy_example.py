import numpy as np
import itertools
import textwrap

def evaluate_fraction(P_mat: np.ndarray,
                      H_mat: np.ndarray,
                      c_strategy: np.ndarray,
                      pi_vec: np.ndarray):
    numerator = 0.0
    denominator = 0.0
    for j, c_idx in enumerate(c_strategy):
        numerator += pi_vec[j] * P_mat[c_idx, j]
        denominator += pi_vec[j] * H_mat[c_idx, j]
    return numerator, denominator
    

def find_optimal_control(P_mat: np.ndarray,
                         H_mat: np.ndarray,
                         pi_vec: np.ndarray):

    # Check dimension consistency
    if not P_mat.shape[1] == H_mat.shape[1] == len(pi_vec):
        raise ValueError("Ambient condition dimensions mismatch.")
    if not P_mat.shape[0] == H_mat.shape[0]:
        raise ValueError("Control input dimensions mismatch.")
    
    num_ac = P_mat.shape[1]   # number of ambient conditions
    num_c = P_mat.shape[0]    # number of possible control inputs

    # Result array with one axis per ambient condition
    max_output = 0
    optimal_control = None
    # Iterate over all possible combinations of control indices
    for c_strategy in itertools.product(range(num_c), repeat=num_ac):
        numerator, denominator = evaluate_fraction(P_mat=P_mat,
                                                   H_mat=H_mat,
                                                   c_strategy=c_strategy,
                                                   pi_vec=pi_vec)
        
        # Store the expected value E(c) for this combination
        output = numerator / denominator if denominator != 0 else np.nan
        if output > max_output:
            optimal_control = c_strategy
            max_output = output

    return optimal_control

def evaluate_control(P_mat: np.ndarray,
                     H_mat: np.ndarray,
                     pi_vec: np.ndarray,
                     c_strategy: np.ndarray,
                     health_budget: float):
    numerator, denominator = evaluate_fraction(P_mat=P_mat,
                                               H_mat=H_mat,
                                               c_strategy=c_strategy,
                                               pi_vec=pi_vec)
    lifetime = health_budget / denominator
    lifetime_energy = lifetime * numerator
    return lifetime, lifetime_energy

def setting_info(pi_vec: np.ndarray):
    
    out_str = f"Ambient conditions prevalence: {pi_vec}\n"
    if np.count_nonzero(pi_vec) == 1:
        greedy_control = np.nonzero(pi_vec)[0][0]
    else:
        greedy_control = None
    
    if greedy_control is not None:
        out_str += f"(corresponds to greedy control for ambient condition n = {greedy_control})"
    else:
        out_str += f"(corresponds to joint optimization problem)"
    return out_str

def control_info(c_strategy: np.ndarray,
                       pi_vec: np.ndarray):
    out_str = ""
    for ac, (c, pi) in enumerate(zip(c_strategy, pi_vec)):
        out_str += f"Control for ambient condition {ac} (Prevalence {pi}): {c}\n"
    return out_str

def result_info(lifetime: float, lifetime_energy: float):

    return f"Resulting lifetime: {lifetime}\nResulting lifetime energy produced: {lifetime_energy}"


P_mat = np.array([[0., 0.],
                  [1., 5.]])
H_mat = np.array([[1., 1.],
                  [2., 3.]])

health_budget = 100 # health budget


tab = "  "
####################################
# greedy control strategies
for pi_vec in [np.array([1, 0]), np.array([0, 1])]:
    print("--------------------")
    print(setting_info(pi_vec=pi_vec))
    optimal_control = \
        find_optimal_control(P_mat=P_mat,
                            H_mat=H_mat,
                            pi_vec=pi_vec)
    print(tab + "Optimal control settings:")
    print(textwrap.indent(control_info(c_strategy=optimal_control, pi_vec=pi_vec), 2 * tab))
    lifetime, lifetime_energy = evaluate_control(P_mat=P_mat,
                                                H_mat=H_mat,
                                                pi_vec=pi_vec,c_strategy=optimal_control,
                                                health_budget=health_budget)
    print(textwrap.indent(result_info(lifetime=lifetime, lifetime_energy=lifetime_energy), tab))

####################################
# Joint optimization
print("--------------------")
pi_vec = np.array([0.5, 0.5])
print(setting_info(pi_vec=pi_vec))
optimal_control = \
    find_optimal_control(P_mat=P_mat,
                        H_mat=H_mat,
                        pi_vec=pi_vec)
print(tab + "Optimal control settings:")
print(textwrap.indent(control_info(c_strategy=optimal_control, pi_vec=pi_vec), 2 * tab))
lifetime, lifetime_energy = evaluate_control(P_mat=P_mat,
                                            H_mat=H_mat,
                                            pi_vec=pi_vec,c_strategy=optimal_control,
                                            health_budget=health_budget)
print(textwrap.indent(result_info(lifetime=lifetime, lifetime_energy=lifetime_energy), tab))

####################################
# Applying greedy result to joint problem
print("--------------------")
print(setting_info(pi_vec=pi_vec))
greedy_control = [1, 1]
print(tab + "Greedy control settings:")
print(textwrap.indent(control_info(c_strategy=greedy_control, pi_vec=pi_vec), 2 * tab))
lifetime, lifetime_energy = evaluate_control(P_mat=P_mat,
                                            H_mat=H_mat,
                                            pi_vec=pi_vec,c_strategy=greedy_control,
                                            health_budget=health_budget)
print(textwrap.indent(result_info(lifetime=lifetime, lifetime_energy=lifetime_energy), tab))

####################################
# Optimal control for extreme scenario
P_mat = np.array([[0., 0.],
                  [1., 5.]])
H_mat = np.array([[0.1, 0.1],
                  [2., 3.]])

print("--------------------")
pi_vec = np.array([0.95, 0.05])
print(setting_info(pi_vec=pi_vec))
optimal_control = \
    find_optimal_control(P_mat=P_mat,
                        H_mat=H_mat,
                        pi_vec=pi_vec)
print(tab + "Optimal control settings:")
print(textwrap.indent(control_info(c_strategy=optimal_control, pi_vec=pi_vec), 2 * tab))
lifetime, lifetime_energy = evaluate_control(P_mat=P_mat,
                                            H_mat=H_mat,
                                            pi_vec=pi_vec,c_strategy=optimal_control,
                                            health_budget=health_budget)
print(textwrap.indent(result_info(lifetime=lifetime, lifetime_energy=lifetime_energy), tab))

####################################
# Applying greedy result to joint problem (extreme scenario)
print("--------------------")
print(setting_info(pi_vec=pi_vec))
greedy_control = [1, 1]
print(tab + "Greedy control settings:")
print(textwrap.indent(control_info(c_strategy=greedy_control, pi_vec=pi_vec), 2 * tab))
lifetime, lifetime_energy = evaluate_control(P_mat=P_mat,
                                            H_mat=H_mat,
                                            pi_vec=pi_vec,c_strategy=greedy_control,
                                            health_budget=health_budget)
print(textwrap.indent(result_info(lifetime=lifetime, lifetime_energy=lifetime_energy), tab))
