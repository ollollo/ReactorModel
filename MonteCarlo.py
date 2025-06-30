import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns

# Модель реактора
def pfr_model(z, y, params):
    C3H6, O2, Acr, AA, CO, CO2, AcOH, H2O, T = y
    R = 8.314

    # Константы Аррениуса
    k1 = params['k1'] * np.exp(-params['E1'] / (R * T))
    k2 = params['k2'] * np.exp(-params['E2'] / (R * T))
    k3 = params['k3'] * np.exp(-params['E3'] / (R * T))
    k4 = params['k4'] * np.exp(-params['E4'] / (R * T))
    k5 = params['k5'] * np.exp(-params['E5'] / (R * T))

    # Скорости реакций
    r1 = k1 * C3H6 * O2
    r2 = k2 * C3H6 * O2**1.5
    r3 = k3 * C3H6 * O2**3
    r4 = k4 * C3H6 * O2**4.5
    r5 = k5 * C3H6 * O2**2.5

    dC3H6_dz = -(r1 + r2 + r3 + r4 + r5) / params['v']
    dO2_dz   = -(r1 + 1.5*r2 + 3*r3 + 4.5*r4 + 2.5*r5) / params['v']
    dAcr_dz  = r1 / params['v']
    dAA_dz   = r2 / params['v']
    dCO_dz   = 3*r3 / params['v']
    dCO2_dz  = 3*r4 / params['v'] + r5 / params['v']
    dAcOH_dz = r5 / params['v']
    dH2O_dz  = (r1 + r2 + 3*r3 + 3*r4 + r5) / params['v']
    dT_dz    = 0

    return [dC3H6_dz, dO2_dz, dAcr_dz, dAA_dz, dCO_dz, dCO2_dz, dAcOH_dz, dH2O_dz, dT_dz]

# Запуск одной симуляции
def run_sim_monte_carlo(T0, ratio, params):
    C3H6_0 = 1.0
    O2_0 = C3H6_0 * ratio
    y0 = [C3H6_0, O2_0, 0, 0, 0, 0, 0, 0, T0]

    sol = solve_ivp(lambda z, y: pfr_model(z, y, params), [0, 10], y0)
    if sol.success:
        y = sol.y
        C3H6_final = y[0, -1]
        Acr_final = y[2, -1]
        AA_final = y[3, -1]
        AcOH_final = y[6, -1]

        conv = (C3H6_0 - C3H6_final) / C3H6_0
        sel_AA = AA_final / (C3H6_0 - C3H6_final) if C3H6_0 != C3H6_final else 0
        sel_Acr = Acr_final / (C3H6_0 - C3H6_final) if C3H6_0 != C3H6_final else 0
        return AA_final, sel_AA, conv, sel_Acr, AcOH_final
    else:
        return 0, 0, 0, 0, 0
# Базовые параметры
base_params = {
    'k1': 1e2, 'E1': 80000,
    'k2': 5e1, 'E2': 90000,
    'k3': 2e1, 'E3': 95000,
    'k4': 1e1, 'E4': 100000,
    'k5': 3e1, 'E5': 97000,
    'v': 0.1
}

np.random.seed(0)
N = 200
results = []

for i in range(N):
    T = np.random.normal(720, 5)
    ratio = np.random.normal(2.0, 0.1)

    perturbed_params = base_params.copy()
    for key in ['k1', 'k2', 'k3', 'k4', 'k5']:
        perturbed_params[key] *= np.random.normal(1.0, 0.1)

    AA, sel, conv, sel_acr, AcOH = run_sim_monte_carlo(T, ratio, perturbed_params)
    results.append({
        'T': T, 'ratio': ratio, 'yield_AA': AA, 'sel_AA': sel,
        'conv_C3H6': conv, 'sel_Acr': sel_acr, 'yield_AcOH': AcOH
    })

df_mc = pd.DataFrame(results)

# Распределения
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
sns.histplot(df_mc['yield_AA'], kde=True, ax=axs[0, 0], color='skyblue')
axs[0, 0].set_title('Выход акриловой кислоты')

sns.histplot(df_mc['sel_AA'], kde=True, ax=axs[0, 1], color='lightgreen')
axs[0, 1].set_title('Селективность по акриловой кислоте')

sns.histplot(df_mc['conv_C3H6'], kde=True, ax=axs[1, 0], color='salmon')
axs[1, 0].set_title('Конверсия пропилена')

sns.histplot(df_mc['yield_AcOH'], kde=True, ax=axs[1, 1], color='orchid')
axs[1, 1].set_title('Выход уксусной кислоты')

plt.tight_layout()
plt.show()

# Корреляции
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

sns.scatterplot(x='T', y='yield_AA', data=df_mc, ax=axs[0, 0])
axs[0, 0].set_title('Температура vs Выход AA')

sns.scatterplot(x='ratio', y='yield_AA', data=df_mc, ax=axs[0, 1])
axs[0, 1].set_title('O2:C3H6 vs Выход AA')

sns.scatterplot(x='T', y='sel_AA', data=df_mc, ax=axs[1, 0])
axs[1, 0].set_title('Температура vs Селективность AA')

sns.scatterplot(x='ratio', y='sel_AA', data=df_mc, ax=axs[1, 1])
axs[1, 1].set_title('O2:C3H6 vs Селективность AA')

plt.tight_layout()
plt.show()
