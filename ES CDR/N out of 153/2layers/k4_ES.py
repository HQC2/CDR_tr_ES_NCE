# %%
import pyscf
import slowquant.SlowQuant as sq
from qiskit_nature.second_q.mappers import JordanWignerMapper
from slowquant.qiskit_interface.interface import QuantumInterface
from slowquant.qiskit_interface.wavefunction import WaveFunction
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
#from slowquant.qiskit_interface.linear_response.projected import quantumLR

from qiskit_ibm_runtime.fake_provider import FakeTorino

from qiskit.primitives import Estimator

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import Sampler
from qiskit_aer.primitives import SamplerV2

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.optimize import minimize
import itertools
import random
from scipy.optimize import curve_fit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import matplotlib.pyplot as plt
import heapq

# %%
mol = pyscf.M(atom="""H 0.0 0.0 0.0;
                      H 1.5 0.0 0.0;
                      H 0.0 1.8 0.0;
                      H 1.5 1.8 0.0;""", basis="sto-3g", unit="angstrom")
rhf = pyscf.scf.RHF(mol).run()

sampler = Estimator()
primitive = sampler
mapper = JordanWignerMapper()
# For H4 you can make the wavefunction better by increasing n_layers.
# n_layers: 3 will prob. give almost the FCI solution.
QI = QuantumInterface(primitive, "tUPS", mapper, ansatz_options={"n_layers": 2, "do_pp": True}, ISA = True)

WF = WaveFunction(
    mol.nao * 2,
    mol.nelectron,
    (4, 4),
    np.load("orbitals.npy"),
    mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
    mol.intor("int2e"),
    QI,
    #include_active_kappa = True
)

#WF.run_vqe_2step("RotoSolve", maxiter=3)
#WF.run_vqe_2step("SLSQP", maxiter=40)
WF.ansatz_parameters = [3.139971979175481, 2.794423289836064, 3.1396666774542217, -3.1279628624838636, -0.3477597864402062, 3.131254822687457, -1.5712922633570714, 0.049457657628517115, 3.1428145974375497, 2.7060354369343496, -0.08622479525081332, 2.7900642540466234, -2.7604987363719484, -0.03332307014761141, -2.7370553249708895, -3.2725379283962637, -0.15194244920006572, 3.2618598820421307]
optimized_parameters = WF.ansatz_parameters
#no noise
nonclif_ground_state_energy = WF.energy_elec
print("Non-Clifford Ground state energy:", nonclif_ground_state_energy)

#device noise
# Update the primitive with simulated noise
backend = FakeTorino()
QI.pass_manager = generate_preset_pass_manager(3,backend=backend,seed_transpiler=123) # seeded standard transpiler
QI.do_postselection = False
QI.do_M_mitigation = False
noise_model = NoiseModel.from_backend(backend)
sampler = Sampler(backend_options={"noise_model":noise_model})
WF.change_primitive(sampler)    
# Calculate the ground state energy using the noisy simulator
noisy_nonclif_ground_state_energy = WF.energy_elec
print("Noisy Non-Clifford Ground state energy:",noisy_nonclif_ground_state_energy)

# %%
linear_results = []
nonlinear_results = []
N = 153
for repeat in range(20):
    #prepare training set
    clif_ground_state_energies = []
    noisy_clif_ground_state_energies = []
    n_replacements = 4
    numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    all_combinations = list(itertools.combinations(numbers, n_replacements))
    selected_combinations = random.sample(all_combinations, N)
    for n in range(N):
        QI = QuantumInterface(primitive, "tUPS", mapper, ansatz_options={"n_layers": 2, "do_pp": True}, ISA = True)

        WF = WaveFunction(
        mol.nao * 2,
        mol.nelectron,
        (4, 4),
        np.load("orbitals.npy"),
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
        #include_active_kappa = True
        )
        WF.ansatz_parameters = [3.141592653589793, 0, 3.141592653589793, -3.141592653589793, 0, 3.141592653589793, -1.5707963267948966, 0, 3.141592653589793, -3.141592653589793, 0, 3.141592653589793, 3.141592653589793, 0, 3.141592653589793, -3.141592653589793, 0, 3.141592653589793]
        indices_to_replace = selected_combinations[n]
        for index in indices_to_replace:
            WF.ansatz_parameters[index] = optimized_parameters[index]
        print(WF.ansatz_parameters)

        clif_ground_state_energy = WF.energy_elec
        clif_ground_state_energies.append(clif_ground_state_energy)
        
        backend = FakeTorino()
        QI.pass_manager = generate_preset_pass_manager(3,backend=backend,seed_transpiler=123) # seeded standard transpiler
        QI.do_postselection = False
        QI.do_M_mitigation = False
        noise_model = NoiseModel.from_backend(backend)
        sampler = Sampler(backend_options={"noise_model":noise_model})
        WF.change_primitive(sampler)   
        noisy_clif_ground_state_energy = WF.energy_elec
        noisy_clif_ground_state_energies.append(noisy_clif_ground_state_energy)

    results_linear = []
    results_nonlinear = []

    for n in range(N):
        i = n + 1
        smallest_clif_ground_state_energies_indices = heapq.nsmallest(i, enumerate(clif_ground_state_energies), key=lambda x: x[1])
        smallest_clif_ground_state_energies = [value for index, value in smallest_clif_ground_state_energies_indices]
        indices = [index for index, value in smallest_clif_ground_state_energies_indices]
        smallest_noisy_clif_ground_state_energies = [noisy_clif_ground_state_energies[i] for i in indices]
        print(smallest_clif_ground_state_energies)
        print(smallest_noisy_clif_ground_state_energies)

        #data preparation
        nonclif_ground_state_energies = [nonclif_ground_state_energy for _ in range(len(smallest_clif_ground_state_energies))]
        noisy_nonclif_ground_state_energies = [noisy_nonclif_ground_state_energy for _ in range(len(smallest_noisy_clif_ground_state_energies))]
        X = np.array(smallest_noisy_clif_ground_state_energies).reshape(-1, 1)
        y = np.array(smallest_clif_ground_state_energies)
        X_test = np.array(noisy_nonclif_ground_state_energies).reshape(-1, 1)
        y_test = np.array(nonclif_ground_state_energies)

        # Linear regression using sklearn
        X_linear = np.column_stack([X.flatten(), np.ones_like(X.flatten())])
        model_linear = LinearRegression().fit(X_linear, y)
        X_test_linear = np.array([[noisy_nonclif_ground_state_energy, 1.0]])
        y_pred_linear = model_linear.predict(X_test_linear)[0]
        results_linear.append(y_pred_linear)
        print('Result of Linear Regression:', y_pred_linear)

        if i >= 3:
            # Nonlinear regression using sklearn as quadratic fit
            X_nonlinear = np.column_stack([
                X.flatten() ** 2,
                X.flatten(),
                np.ones_like(X.flatten())
            ])
            model_nonlinear = LinearRegression().fit(X_nonlinear, y)
            X_test_nonlinear = np.array([[noisy_nonclif_ground_state_energy ** 2, noisy_nonclif_ground_state_energy, 1.0]])
            y_pred_nonlinear = model_nonlinear.predict(X_test_nonlinear)[0]
            results_nonlinear.append(y_pred_nonlinear)
            print('Result of Nonlinear Regression:', y_pred_nonlinear)
    linear_results.append(results_linear)
    nonlinear_results.append(results_nonlinear)

print("Linear Regression Results:", linear_results)
print("Nonlinear Regression Results:", nonlinear_results)

# %%
import csv
def save_to_csv(filename, data_dict):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Type", "Results"])
        for key, values in data_dict.items():
            for value in values:
                writer.writerow([key, value])

save_to_csv("2layers_ES.csv", {
    "Linear": linear_results,
    "Nonlinear": nonlinear_results,
})

print("Data saved to 2layers_ES.csv")



