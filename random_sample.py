
import numpy as np
import matplotlib.pyplot as plt

RHO_FE = 7000 #liquid Fe density [kg m-3]
RHO_FEO = 4400 #liquid FeO density [kg m-3]


args_array = np.loadtxt('./Melt_Scenario2/oxi_standard/example_co2_15/args_array.dat')


masses    = args_array[:, 0]  # kg
speeds    = args_array[:, 1]  # [m/s]
angles    = args_array[:, 2]  # [rad]

radii_m  = (3 * masses / (4 * np.pi * RHO_FE))**(1/3) #m
diameters = radii_m * 1e6 * 2  # µm

fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=140)

ax = axes[0]
ax.hist(diameters, bins=20, density=True, histtype='step',
        color='steelblue', linewidth=1.8, label='Samples')
ax.set_xlabel('Initial diameter (µm)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('(a) Initial diameter', fontsize=12)
ax.text(0.9, 0.97, f'mean = {masses.mean()*1e9:.1f} µg\n'
                     f'std  = {masses.std()*1e9:.1f} µg',
                    transform=ax.transAxes, va='top', ha='right', fontsize=10)

ax = axes[1]
ax.hist(speeds / 1e3, bins=20, density=True, histtype='step',
        color='coral', linewidth=1.8)
ax.set_xlabel('Entry speed (km/s)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('(b) Entry speed', fontsize=12)
ax.text(0.97, 0.97, f'mean = {speeds.mean()/1e3:.1f} km/s\n'
                     f'std  = {speeds.std()/1e3:.1f} km/s',
        transform=ax.transAxes, va='top', ha='right', fontsize=10)
ax = axes[2]
ax.hist(np.degrees(angles), bins=20, density=True, histtype='step',
        color='seagreen', linewidth=1.8)
ax.set_xlabel('Entry angle (°)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('(c) Entry angle', fontsize=12)
ax.text(0.97, 0.97, f'mean = {np.degrees(angles).mean():.1f}°\n'
                     f'std  = {np.degrees(angles).std():.1f}°',
        transform=ax.transAxes, va='top', ha='right', fontsize=10)

for ax in axes:
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

data = np.column_stack([
    diameters,           # µm
    speeds / 1e3,        # km/s
    np.degrees(angles)   # rad
])