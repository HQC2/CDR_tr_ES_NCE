# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed


# %%

nonclif_ground_state_energy = -3.712209125939833
results = []
std = []
avg = []
deltas = []

for m in range(17):
    N = m + 1
    noisy_value = -3.1034797862655883
    k_fixed = 18

    def process_i(i):
        df = pd.read_csv('k_from_1_to_9.csv')
        df_filtered = df[(df['k'] >= 1) & (df['k'] <= N) & (df['t'] == i)]

        def sample_func(group):
            if len(group) <= 1000:
                return group
            else:
                return group.sample(n=1000)

        df_sampled = df_filtered.groupby('k', group_keys=False).apply(sample_func)

        X_noisy = df_sampled['x_noisy'].values
        X_exact = df_sampled['x_exact'].values
        k = df_sampled['k'].values

        X_features = np.column_stack([
            X_noisy**2,
            k * X_noisy,
            k**2,
            k,
            X_noisy,
            np.ones_like(X_noisy)
        ])

        model = LinearRegression()
        model.fit(X_features, X_exact)

        X_input = np.array([
            noisy_value**2,
            k_fixed * noisy_value,
            k_fixed**2,
            k_fixed,
            noisy_value,
            1.0
        ]).reshape(1, -1)

        y_pred = model.predict(X_input)[0]
        return y_pred

    regression = Parallel(n_jobs=20)(delayed(process_i)(i) for i in range(20))

    std_dev = np.std(regression)
    average = np.mean(regression)
    delta = np.abs(nonclif_ground_state_energy - average)

    std.append(std_dev)
    avg.append(average)
    deltas.append(delta)
    results.append(regression)

# %%
df = pd.DataFrame({
    'delta': deltas,
    'std': std
})

df.to_csv('NCE_kscan.csv', index=False)

