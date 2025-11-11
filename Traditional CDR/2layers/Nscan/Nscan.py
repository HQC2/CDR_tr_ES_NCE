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

import matplotlib.pyplot as plt
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
def run_single_repeat(repeat):
    clif_ground_state_energies = []
    noisy_clif_ground_state_energies = []
    results_linear = []
    results_nonlinear = []
    
    n_replacements = 4
    numbers = list(range(18))
    all_combinations = list(itertools.combinations(numbers, n_replacements))

    if len(all_combinations) < 300:
        raise ValueError(f"Cannot sample 300 from only {len(all_combinations)} combinations")
        
    selected_combinations = random.sample(all_combinations, 300)

    for n in range(300):
        QI = QuantumInterface(primitive, "tUPS", mapper, ansatz_options={"n_layers": 2, "do_pp": True}, ISA=True)

        WF = WaveFunction(
            mol.nao * 2,
            mol.nelectron,
            (4, 4),
            np.load("orbitals.npy"),
            mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
            mol.intor("int2e"),
            QI,
        )
        WF.ansatz_parameters = [
            3.141592653589793, 0, 3.141592653589793, -3.141592653589793, 0, 3.141592653589793,
            -1.5707963267948966, 0, 3.141592653589793, -3.141592653589793, 0, 3.141592653589793,
            3.141592653589793, 0, 3.141592653589793, -3.141592653589793, 0, 3.141592653589793
        ]
        indices_to_replace = selected_combinations[n]
        for index in indices_to_replace:
            WF.ansatz_parameters[index] = optimized_parameters[index]
        print(WF.ansatz_parameters)

        clif_ground_state_energy = WF.energy_elec
        clif_ground_state_energies.append(clif_ground_state_energy)

        backend = FakeTorino()
        QI.pass_manager = generate_preset_pass_manager(3, backend=backend, seed_transpiler=123)
        QI.do_postselection = False
        QI.do_M_mitigation = False
        noise_model = NoiseModel.from_backend(backend)
        sampler = Sampler(backend_options={"noise_model": noise_model})
        WF.change_primitive(sampler)
        noisy_clif_ground_state_energy = WF.energy_elec
        noisy_clif_ground_state_energies.append(noisy_clif_ground_state_energy)

        nonclif_ground_state_energies = [nonclif_ground_state_energy] * len(clif_ground_state_energies)
        noisy_nonclif_ground_state_energies = [noisy_nonclif_ground_state_energy] * len(noisy_clif_ground_state_energies)

        X = np.array(noisy_clif_ground_state_energies).reshape(-1, 1)
        y = np.array(clif_ground_state_energies)
        X_test = np.array(noisy_nonclif_ground_state_energies).reshape(-1, 1)
        y_test = np.array(nonclif_ground_state_energies)

        # Linear regression
        def cost_function(a, X_exact, X_noisy):
            a1, a2 = a
            return np.sum((X_exact - (a1 * X_noisy + a2))**2)

        X_exact = np.array(clif_ground_state_energies)
        X_noisy = np.array(noisy_clif_ground_state_energies)
        initial_guess = [1, 0]
        result = minimize(cost_function, initial_guess, args=(X_exact, X_noisy))
        a1_optimal, a2_optimal = result.x
        y_pred_linear = a1_optimal * noisy_nonclif_ground_state_energy + a2_optimal
        print('result of square regression:', y_pred_linear)
        results_linear.append(y_pred_linear)

        if n > 1:
            def nonlinear_model(X_noisy, a1, a2, a3):
                return a1 * X_noisy**2 + a2 * X_noisy + a3

            X_exact = np.array(clif_ground_state_energies)
            X_noisy = np.array(noisy_clif_ground_state_energies)
            initial_guess = [1, 1, 0]
            params, covariance = curve_fit(nonlinear_model, X_noisy, X_exact, p0=initial_guess)
            a1_optimal, a2_optimal, a3_optimal = params
            y_pred_nonlinear = nonlinear_model(noisy_nonclif_ground_state_energy, a1_optimal, a2_optimal, a3_optimal)
            print('Result of Nonlinear Fitting:', y_pred_nonlinear)
            results_nonlinear.append(y_pred_nonlinear)
        else:
            print('number of samples is not enough')

    return results_linear, results_nonlinear

results = Parallel(n_jobs=20)(delayed(run_single_repeat)(i) for i in range(20))

linear_results, nonlinear_results = zip(*results)
linear_results = list(linear_results)
nonlinear_results = list(nonlinear_results)

# %%
import csv
def save_to_csv(filename, data_dict):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Type", "Results"])
        for key, values in data_dict.items():
            for value in values:
                writer.writerow([key, value])

save_to_csv("impact of N_biased.csv", {
    "Linear": linear_results,
    "Nonlinear": nonlinear_results,
})

print("Data saved to impact of N_biased.csv")


