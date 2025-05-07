import math
import cvxpy as cp
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

def init_param_hetero(constant, n, t):
    parameters = {'sigma' : constant['sigma'], 'D_n': np.empty(n), 'Gamma': np.empty(n), 'local_iter': np.empty(n), 'c_n': np.empty(n),
                  'frequency_n' : np.empty(n), 'weight_size_n' : np.empty(n),
                  'number_of_clients' : n, 'bandwidth' : np.empty(n), 'channel_gain_n': np.empty(n), 'transmission_power_n' : np.empty(n), 'noise_W' : np.empty(n),
                  'transmission_rate' : np.empty(n), 't': t}

    # Scaling factors for unit conversion
    scale_bandwidth = 1e6  # Convert MHz to Hz
    scale_MB_to_bit = 8 * 1e6 # Convert MB to bit
    
    for i in range(constant['number_of_clients']):
        # d_i = random.randrange(0, 1) # 0만 나옴
        f_i = random.randrange(0, 1)
        t_i = random.randrange(0, 1)
        parameters["sigma"] = constant["sigma"]
        parameters["D_n"][i] = constant["D_n"][i]
        parameters["Gamma"][i] = constant["Gamma"]
        parameters["local_iter"][i] = constant["local_iter"]
        parameters["c_n"][i] = constant["c_n"]
        parameters["frequency_n"][i] = constant["frequency_n_GHz"][f_i]
        parameters["weight_size_n"][i] = constant["weight_size_n_kbit"]

        # Calculate R
        parameters["number_of_clients"] = constant["number_of_clients"]
        parameters["bandwidth"][i] = constant["bandwidth_MHz"]
        parameters["channel_gain_n"][i] = constant["channel_gain_n"]
        parameters["transmission_power_n"][i] = constant["transmission_power_n"][t_i]
        parameters["noise_W"][i] = constant["noise_W"]
        parameters["transmission_rate"][i] = parameters["bandwidth"][i]/parameters["number_of_clients"] \
            * math.log(1 + parameters["channel_gain_n"][i] * parameters["transmission_power_n"][i] / parameters["noise_W"][i])

    return parameters


def objective_function(v_n, t, r, parameters):
    objective = -0.00000001*parameters['sigma']**(r-1) * cp.sum(v_n @ parameters['D_n']) / t
    return objective

def result_function(v_n, t, r, parameters):
    objective = -parameters['sigma']**(r-1) * \
        sum(v_n[i] * parameters['D_n'][i] for i in range(parameters["number_of_clients"])) / t
    return objective

def block_coordinate_descent(parameters, round, t):
    n = parameters['number_of_clients']

    # Define variables
    v_n = cp.Variable(n)
    # t = cp.Variable()

    # Define constraints for each block
    constraints = []
    for i in range(n):
        block_constraints = [
            parameters['Gamma'] <= v_n[i],
            v_n[i] <= 1,
            parameters['local_iter'][i] * parameters['c_n'][i] * v_n[i] * parameters['D_n'][i] /\
                 parameters['frequency_n'][i] + parameters['weight_size_n'][i] /\
                     parameters['transmission_rate'][i] <= t
        ]
        constraints += block_constraints
    
    t_optimal = t
    # Block coordinate descent 반복
    max_iter = 50
    tolerance = 1e-12
    pre_sol = 9999999
    sol_list = []

    # Solve the optimization problem using block coordinate descent
    for _ in range(max_iter):  # Set the desired number of iterations
        v_n_objective = cp.Minimize(objective_function(v_n, t_optimal, round, parameters))
        v_n_problem = cp.Problem(v_n_objective, constraints)
        v_n_problem.solve(qcp=True, solver=cp.ECOS)
        v_n_optimal = v_n.value
        max_t = []
        for i in range(n):
            tmp = [parameters['local_iter'][i] * parameters['c_n'][i] * v_n_optimal[i] *\
                    parameters['D_n'][i] / parameters['frequency_n'][i] + \
                        parameters['weight_size_n'][i] / parameters['transmission_rate'][i]]
            max_t += tmp

        t_optimal = max(max_t)
        sol = result_function(v_n_optimal, t_optimal, round, parameters)

    sol_list.append(sol)
    v_n_optimal = [float(x) for x in v_n_optimal]

    return v_n_optimal, sol_list, t_optimal

def descent_01(parameters, round, t):
    n = parameters['number_of_clients']
    v_n = [1.0 for _ in range(n)]
    
    t_optimal = t
    # Block coordinate descent 반복
    max_iter = 1
    sol_list = []

    # Solve the optimization problem using block coordinate descent
    for r in tqdm(range(round)):
        for _ in range(max_iter):  # Set the desired number of iterations
            # print(v_n.value)
            # y에 대한 최적화 (x를 상수로 고정)
            max_t = []
            for i in range(n):
                tmp = [parameters['local_iter'][i] * parameters['c_n'][i] * v_n[i] *\
                        parameters['D_n'][i] / parameters['frequency_n'][i] + \
                            parameters['weight_size_n'][i] / parameters['transmission_rate'][i]]
                max_t += tmp

            t_optimal = max(max_t)
            sol = result_function(v_n, t_optimal, r)

        # print(sol)
        sol_list.append(sol)
        if r % 15 == 0:
            v_n = [x -0.1 for x in v_n]


    return sol_list

if __name__ == "__main__":
    # Example usage
    constant_parameters = {'sigma' : 0.8, 'D_n': [2500], 'Gamma': 0.1, 'local_iter': 10, 'c_n': 3*1e4,
                    #   'frequency_n_GHz' : [1.5, 2, 2.5, 3], 
                    'frequency_n_GHz' : [1.5], 
                    'weight_size_n_kbit' : 100,
                    'number_of_clients' : 5, 'bandwidth_MHz' : 1, 'channel_gain_n': 1, 
                    #   'transmission_power_n' : [0.2, 0.5, 1], 
                    'transmission_power_n' : [1], 
                    'noise_W' : 1e-12}

    parameters = init_param_hetero(constant_parameters, constant_parameters['number_of_clients'])
    # parameters
    # print(parameters)

    r = 1  # You need to define the value of r
    t = parameters['t']
    # block_coordinate_descent(parameters, r, t)
    optimal_v_n, sol_list, optimal_t = block_coordinate_descent(parameters, r, t)
    print(optimal_v_n)
    print(optimal_t)
    # sol_list_01 = descent_01(parameters, r, t)

    # print("Optimal vn:", optimal_v_n)
    # print("Optimal t:", optimal_t)
