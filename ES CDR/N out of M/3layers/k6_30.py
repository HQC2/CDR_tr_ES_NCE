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

import numpy as np
from scipy.optimize import minimize
import itertools
import random
from scipy.optimize import curve_fit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import heapq
from joblib import Parallel, delayed
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
QI = QuantumInterface(primitive, "tUPS", mapper, ansatz_options={"n_layers": 3, "do_pp": True}, ISA = True)

WF = WaveFunction(
    mol.nao * 2,
    mol.nelectron,
    (4, 4),
    np.load("orbitals_3layers.npy"),
    mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
    mol.intor("int2e"),
    QI,
    include_active_kappa = True
)

#WF.run_vqe_2step("RotoSolve", maxiter=3)
#WF.run_vqe_2step("SLSQP", maxiter=40)
WF.ansatz_parameters = [3.1417031379418052, 2.7625992578966563, 3.1417334893610525, 3.141585064865313, 2.7972128361572564, 3.141645464656413, 1.5707640394797453, 0.1567972368842604, 3.141771986548728, -3.1393134115102455, 0.1311032601149695, -2.867353761271364, 0.38162729013715574, -0.1980287652881329, -3.247758074235468, 3.1412212373128066, -0.20963353637741186, 3.1422000595045434, 0.35370738464222384, -0.061450985229703804, 3.251018603529601, -2.9318597256025525, 0.0808917950196615, 0.27181169190183185, 3.141445836663981, -0.029510697601092685, 3.1414893486856466]
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
def run_single_repeat(repeat):
    random.seed(repeat) 
    clif_ground_state_energies = []
    noisy_clif_ground_state_energies = []
    results_linear = []
    results_nonlinear = []
    N = 30
    n_replacements = 6
    numbers = list(range(27))
    all_combinations = list(itertools.combinations(numbers, n_replacements))
    selected_combinations = random.sample(all_combinations, 1000)

    for n in range(1000):
        QI = QuantumInterface(primitive, "tUPS", mapper, ansatz_options={"n_layers": 3, "do_pp": True}, ISA=True)

        WF = WaveFunction(
            mol.nao * 2,
            mol.nelectron,
            (4, 4),
            np.load("orbitals_3layers.npy"),
            mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
            mol.intor("int2e"),
            QI,
        )
        WF.ansatz_parameters = [3.141592653589793, 0, 3.141592653589793, 3.141592653589793, 0, 3.141592653589793, 1.5707963267948966, 0, 3.141592653589793, -3.141592653589793, 0, -3.141592653589793, 0, 0, -3.141592653589793, 3.141592653589793, 0, 3.141592653589793, 0, 0, 3.141592653589793, -3.141592653589793, 0, 0, 3.141592653589793, 0, 3.141592653589793]

        indices_to_replace = selected_combinations[n]
        for index in indices_to_replace:
            WF.ansatz_parameters[index] = optimized_parameters[index]

        clif_energy = WF.energy_elec
        clif_ground_state_energies.append(clif_energy)

        backend = FakeTorino()
        QI.pass_manager = generate_preset_pass_manager(3, backend=backend, seed_transpiler=123)
        QI.do_postselection = False
        QI.do_M_mitigation = False
        noise_model = NoiseModel.from_backend(backend)
        sampler = Sampler(backend_options={"noise_model": noise_model})
        WF.change_primitive(sampler)
        noisy_energy = WF.energy_elec
        noisy_clif_ground_state_energies.append(noisy_energy)

        if n + 1 >= N:
            smallest_indices = heapq.nsmallest(N, enumerate(clif_ground_state_energies), key=lambda x: x[1])
            smallest_clif = [v for i, v in smallest_indices]
            indices = [i for i, v in smallest_indices]
            smallest_noisy_clif = [noisy_clif_ground_state_energies[i] for i in indices]

            nonclif_ground_state_energies = [nonclif_ground_state_energy] * N
            noisy_nonclif_ground_state_energies = [noisy_nonclif_ground_state_energy] * N

            X = np.array(smallest_noisy_clif).reshape(-1, 1)
            y = np.array(smallest_clif)
            X_test = np.array(noisy_nonclif_ground_state_energies).reshape(-1, 1)
            y_test = np.array(nonclif_ground_state_energies)

            # Linear fit using sklearn
            X_linear = np.column_stack([X.flatten(), np.ones_like(X.flatten())])
            model_linear = LinearRegression().fit(X_linear, y)
            X_test_linear = np.array([[noisy_nonclif_ground_state_energy, 1.0]])
            y_pred_linear = model_linear.predict(X_test_linear)[0]
            results_linear.append(y_pred_linear)

            # Nonlinear model as linear regression on x^2, x, 1
            X_nonlinear = np.column_stack([
                X.flatten()**2,
                X.flatten(),
                np.ones_like(X.flatten())
            ])
            model_nonlinear = LinearRegression().fit(X_nonlinear, y)
            X_test_nonlinear = np.array([
                [noisy_nonclif_ground_state_energy**2, noisy_nonclif_ground_state_energy, 1.0]
            ])
            y_pred_nonlinear = model_nonlinear.predict(X_test_nonlinear)[0]
            results_nonlinear.append(y_pred_nonlinear)

    return results_linear, results_nonlinear

results = Parallel(n_jobs=50)(delayed(run_single_repeat)(repeat) for repeat in range(100))

linear_results, nonlinear_results = zip(*results)

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

save_to_csv("30.csv", {
    "Linear": linear_results,
    "Nonlinear": nonlinear_results,
})


