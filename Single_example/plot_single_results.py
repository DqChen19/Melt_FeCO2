import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import pi

#####
def load_columns(filename):
    df = pd.read_csv(filename, header=None)
    mfe  = np.array(df.iloc[:, 8])
    mfeo = np.array(df.iloc[:, 9])

    rad = ((3*mfe/7000.0 + 3*mfeo/4400.0) / (4.0*pi))**(1/3) * 1e6  # m → µm
    #{vel:.6g}, {temp:.6g}, {(alt-EARTH_RAD)/1000:.6g}, {ox_enc:.3g}, {rate_co2:.3g}, {add_o: .3g}, {add_co2:.3g}, {ram_p:.6g}")
    return {
        'vel':  np.array(df.iloc[:, 0]),
        'temp': np.array(df.iloc[:, 1]),
        'alt':  np.array(df.iloc[:, 2]),
        'oxi':  np.array(df.iloc[:, 3]),
        'kin':  np.array(df.iloc[:, 4]), #rate co2
        'add_o':    np.array(df.iloc[:, 5]),
        'add_co2':  np.array(df.iloc[:, 6]), 
        'dyp':  np.array(df.iloc[:, 7]), 
        'mfe': mfe,
        'mfeo':mfeo,
        'rad': rad
    }

#####
datasets = {
    '20': load_columns('./Single_example/Ver1_Output_CO2_20.log'),
    '70': load_columns('./Single_example/Ver1_Output_CO2_69.log'),
}

alt20 = datasets['20']['alt']
alt70 = datasets['70']['alt']

rad_fe_20  = datasets['20']['mfe']/7000
rad_feo_20 = datasets['20']['mfeo']/4400 
frac_20    = rad_fe_20 ** 2/ (rad_fe_20 ** 2 + rad_feo_20 ** 2)

rad_fe_70  = datasets['70']['mfe']/7000 #((3*datasets['70']['mfe']/7000.0)  / (4.0*pi))**(1/3)
rad_feo_70 = datasets['70']['mfeo']/4400 #((3*datasets['70']['mfeo']/4400.0) / (4.0*pi))**(1/3)
frac_70    = rad_fe_70 ** 2 / (rad_fe_70 ** 2 + rad_feo_70 ** 2)

fig, axs = plt.subplots(3, 3, figsize=(12,9), sharey=False)
plt.subplots_adjust(wspace=0.1, hspace=0.1)

plt.rcParams['font.size'] = 14          
plt.rcParams['axes.labelsize'] = 15     
plt.rcParams['axes.titlesize'] = 15     
plt.rcParams['xtick.labelsize'] = 14   
plt.rcParams['ytick.labelsize'] = 14   
plt.rcParams['legend.fontsize'] = 14   

color20 = 'blue'
color70 = 'green'


for i in range(3):
    for j in range(3):
        axs[i, j].set_ylim(78.4, 92)
        axs[i, j].set_yticks([80,84,88,92])


ax = axs[0, 0]
ax.axvspan(1650, 2400, color='yellow', alpha=0.15, label='Reactive region')
ax.axvspan(1811, 2400, color='red', alpha=0.1, label='Reactive region')
ax.plot(datasets['20']['temp'], alt20, c=color20, label='20 vol% CO$_2$')
ax.plot(datasets['70']['temp'], alt70, c=color70,   label='70 vol% CO$_2$')
ax.axvline(x = 1650, ymin = 0,ymax = 1, ls = "--",color = 'black', alpha = 0.5)
ax.axvline(x = 1811, ymin = 0,ymax = 1, ls = "--",color = 'black', alpha = 0.5)
ax.text(1630, y = 80, s = 'FeO Melt T', rotation = 270, fontsize = 12)
ax.text(1790, y = 80, s = 'Fe Melt T', rotation = 270, fontsize = 12)
ax.set_xlabel('Temperature T (K)', fontsize = 14)
ax.set_ylabel('Altitude (km)',fontsize = 14)
ax.set_title('(A) Temperature profile')
ax.set_xlim([800,2400])
ax.tick_params(axis='both', labelsize=14)
ax.set_xticks([800,1200,1600,2000,2400])


ax = axs[0, 1]
ax.plot(datasets['20']['vel']/1000, alt20, c=color20, label='20 vol% CO$_2$')
ax.plot(datasets['70']['vel']/1000, alt70, c=color70,   label='70 vol% CO$_2$')
ax.set_xlabel('Velocity v (km s$^{-1}$)', fontsize = 14)
ax.set_title('(B) Velocity profile')
ax.set_xlim([0,14])
ax.tick_params(axis='both', labelsize=14)

ax = axs[0, 2]
ax.plot(datasets['20']['rad']*2, alt20, c=color20, label='20 vol% CO$_2$')
ax.plot(datasets['70']['rad']*2, alt70, c=color70,   label='70 vol% CO$_2$')
ax.set_xlabel('Diameter (µm)', fontsize = 14)
ax.set_title('(C) Particle diameter')
ax.tick_params(axis='both', labelsize=14)

#
ax = axs[1, 0]
ax.plot(frac_20, alt20, c=color20, label='20 vol% CO$_2$')
ax.plot(frac_70, alt70, c=color70,   label='70 vol% CO$_2$')
ax.set_xlabel('Fe/(Fe + FeO) in vol', fontsize = 14)
ax.set_ylabel('Altitude (km)', fontsize =14)
ax.set_title('(D) Metallic Fe fraction')
ax.set_xlim([0,1])
ax.tick_params(axis='both', labelsize=14)
ax.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
ax.set_ylabel('Altitude (km)',fontsize = 14)

ax = axs[1, 1]
ax.plot(datasets['20']['mfeo']*1e3/(56+16), alt20, c=color20, label='20 vol% CO$_2$')
ax.plot(datasets['70']['mfeo']*1e3/(56+16), alt70, c=color70,   label='70 vol% CO$_2$')
ax.set_xlabel('FeO (mol)', fontsize = 14)
ax.set_title('(E) Cumulative FeO produced')
ax.tick_params(axis='both', labelsize=14)

ax = axs[1, 2]
ax.plot(datasets['20']['oxi'], alt20, c=color20, label='20 vol% CO$_2$')
ax.plot(datasets['70']['oxi'], alt70, c=color70,   label='70 vol% CO$_2$')
ax.set_xlabel('Oxidation per step (kg/s)', fontsize = 14)
ax.set_title('(F) Total oxidation')
ax.set_xscale('log')
ax.tick_params(axis='both', labelsize=14)


ax = axs[2, 0]
ax.plot(datasets['20']['add_o'], alt20, c=color20, label='20 vol% CO$_2$')
ax.plot(datasets['70']['add_o'], alt70, c=color70, label='70 vol% CO$_2$')
ax.set_xlabel('O uptake (kg/s)', fontsize = 14)
ax.set_title('(G) Incident O ')
ax.set_xscale('log')
ax.tick_params(axis='both', labelsize=14)
ax.set_ylabel('Altitude (km)',fontsize = 14)

ax = axs[2, 1]
ax.plot(np.minimum(datasets['20']['kin'], datasets['20']['add_co2'])/datasets['20']['add_co2'], alt20, c=color20, label='20 vol%  CO$_2$')
ax.plot(np.minimum(datasets['70']['kin'], datasets['70']['add_co2'])/datasets['70']['add_co2'], alt70, c=color70,   label='70 vol% CO$_2$')
ax.tick_params(axis='both', labelsize=14)
ax.set_title('(H) CO$_2$ reacted/incident')
ax.set_xscale('log')
ax.set_xlabel('Ratio', fontsize = 14)

ax = axs[2, 2]
ax.plot(datasets['20']['add_o'] / datasets['20']['oxi'], alt20, c=color20, label='20 vol% total CO$_2$')
ax.plot(datasets['70']['add_o'] / datasets['70']['oxi'], alt70, c=color70,   label='70 vol% total CO$_2$')
ax.set_xlabel('Ratio', fontsize = 14)
ax.tick_params(axis='both', labelsize=14)
ax.set_title('(I) O+O$_2$ reacted/total oxidants')
#ax.set_xscale('log')

ax.set_xticks([0,0.2,0.4,0.6,0.8,1.0])

#ax.set_xlim([2e-12,5e-11])
handles, labels = axs[2,1].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.02),
    ncol=2,
    fontsize=14,
    frameon = False
)

plt.tight_layout(rect=[0,0.05,1,1])

#plt.show()
#fig.savefig('./Single_example/Single_plot_results.pdf')

print(sum(datasets['20']['oxi'][4320:])/sum(datasets['20']['oxi']))
print(sum(datasets['70']['oxi'][4307:])/sum(datasets['70']['oxi']))
