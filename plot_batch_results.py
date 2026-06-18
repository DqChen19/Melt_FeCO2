import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
FE_MELTING_TEMP = 1811
def readModelDataFile(filename):
    """
    Helper function to read model data saved to file.
    Read the data from an output file.

    Inputs:
        filename - the file to read

    Returns:
        result - the data from the file
    """
    
    file_obj = open(filename, "r")
    result = []
    data_not_started = True

    for line in file_obj:
        line_split = line.split()
        #make sure this isn't a label line
        if data_not_started:
            try:
                float(line_split[0])
                data_not_started = False
            except ValueError:
                continue

        if len(line_split) == 1:
            num_val = float(line_split[0])
            result.append(num_val)
        else:
            nums = []
            for num in line_split:
                num_val = float(num)
                nums.append(num_val)
            result.append(tuple(nums))

    return result

def cal_co2_data(val_list, dir,):
    num_runs = len(val_list) #Changeable
    
    means = np.zeros(num_runs)
    co2_percents = np.zeros(num_runs)
    std_tops = np.zeros(num_runs)
    std_bots = np.zeros(num_runs)
    pure_ox_frac = np.zeros(num_runs)
    for i, val in enumerate(val_list):
        
        fname = "/co2_%d/clean_results.dat" % (val)
        results = readModelDataFile(dir + fname)
        particle_fractions = []
        pure_ox_count = 0
        pure_fe_count = 0
 
        for j in range(len(results)):
            frac = results[j][1]
            rad = results[j][0]
            max_temp = results[j][2]

            if  frac < 1 and max_temp > FE_MELTING_TEMP:
                particle_fractions.append(frac)
            if frac == 0:
                pure_ox_count+= 1
            if frac == 1 and max_temp > FE_MELTING_TEMP:
                pure_fe_count += 1
            
        total_ox = len(particle_fractions) 
        pure_ox_frac[i] = pure_ox_count / total_ox
        
        if len(particle_fractions) > 0:
            means[i] = np.mean(particle_fractions)
            std = np.std(particle_fractions)
        else:
            means[i] = 0
            std = 0

        std_tops[i] = means[i] + 2*std
        std_bots[i] = max(0, means[i] - 2 * std)
        co2_percents[i] = val
    return co2_percents, means, std_tops, std_bots, pure_ox_frac

def token(tok):
    if tok == 'standard':
        wus_e = [9,69]
        wus_f = [83.5,100]
        label = None
    elif tok == 'warm':
        wus_e = [8.5,82]
        wus_f = [66.5,100]
        label = 'a) warm'
    elif tok == 'hmeth':
        wus_e = [10,75]
        wus_f = [83,100]
        label = 'b) hCH4'
    elif tok == 'hkzz':
        wus_e = [8,100]
        wus_f = [71,100]
        label = 'c) hkzz'
    else:
        wus_e = [4,59]
        wus_f = [38,100]
        label = 'd) lthre'
    return wus_e, wus_f , label 

def plot_co2_data(ax, dir, dir_list, package):

    val_list = dir_list

    co2_percents, means, std_tops, std_bots, pure_ox_frac = cal_co2_data(val_list,dir)
    f_size = 12
    ax.tick_params(labelsize=f_size)
    ax.set_xlabel(r"Atmospheric CO$_2$ [Vol%]", fontsize=f_size)
    ax.set_ylabel("Fe Fractional Area", fontsize=f_size)

    ax.plot(co2_percents, means, '-', color=package['color'])
    ax.fill_between(co2_percents, std_bots, std_tops, color=package['color'], alpha=0.35, label=package['label'])

    
    if len(pure_ox_frac) > 0 :
        ax_twin = ax.twinx()
        ax_twin.plot(co2_percents, pure_ox_frac, '--', lw = 2, alpha = 0.8, color=package['color'], label='Fully Ox. CO$_2$ summative scenario')
        ax_twin.set_ylim(0, 1)
        ax_twin.set_ylabel("Fully Oxidized\nFraction", fontsize=13, color='green')
        ax_twin.tick_params(axis='y', labelsize=f_size, colors='green')

tomkins_e = 0.726
tomkins_f = 0.024

if __name__ == '__main__':
    fig3 = 1
    if fig3:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=300)
        ax1   = axes[0]  
        ax2   = axes[1]  

    
        dir_1 = np.arange(2, 93, 5)
        package1 = {'color': 'blue', 'label': 'Kinetic'}
        plot_co2_data(ax1, dir="./Melt_Scenario2/oxi_standard", dir_list=dir_1, package=package1)
        

        dir_2 = np.arange(2, 93, 5)
        package2 = {'color': 'purple', 'label': 'Kinetic - extrapolate from solid iron'}
        plot_co2_data(ax1, dir="./Melt_Scenario3/oxi_standard", dir_list=dir_2, package=package2)

        dir_3 = np.arange(2, 93, 5)
        package3 = {'color': 'orange', 'label': 'Add all'}
        plot_co2_data(ax1, dir="./Scenario1/add_standard", dir_list=dir_3, package=package3)

        ax1.axvline(x = 11,  ymin=0.73, ymax=1, c = 'blue', lw = 1.5, alpha = 0.8, ls = '--')
        ax1.plot(11, 0.74,marker='v', color='blue', markersize=5, alpha=0.8, transform=ax1.get_xaxis_transform(), clip_on=False)
        ax1.set_xlim(2, 92)
        ax1.set_ylim(0, 1)
        ax1.set_xticks(np.arange(5, 90, 15))
        ax1.grid(True, linestyle=':', alpha=0.3)
        ax1.set_title('(A) Oxidation > 1650 K')
        ax1.axhline(y = 0.726, c = 'r', lw = 2, alpha = 0.6, ls = ':')
        ax1.tick_params(top=True, labeltop=True)
        ax1.legend(loc='lower left', ncol=1, fontsize=12, frameon=False)
        # 
        dir_1 = np.arange(2, 93, 5)
        package1 = {'color': 'blue', 'label': 'Kinetic'}
        plot_co2_data(ax2, dir="./Melt_Scenario2/oxi_lthre", dir_list=dir_1, package=package1)

        dir_2 = np.arange(2, 93, 5)
        package2 = {'color': 'purple', 'label': 'Kinetic - extrapolate from solid iron'}
        plot_co2_data(ax2, dir="./Melt_Scenario3/oxi_lthre", dir_list=dir_2, package=package2)

        dir_3 = np.arange(2, 93, 5)
        package3 = {'color': 'orange', 'label': 'Add all'}
        plot_co2_data(ax2, dir="./Scenario1/add_lthre", dir_list=dir_3, package=package3)

        ax2.set_xlim(2, 92)
        ax2.set_ylim(0, 1)
        ax2.set_xticks(np.arange(5, 90, 15))
        ax2.tick_params(top=True, labeltop=True)
        ax2.grid(True, linestyle=':', alpha=0.3)
        ax2.set_title('(B) Oxidation > 1000 K')
        ax2.axhline(y = 0.726, c = 'r', lw = 2, alpha = 0.6, ls = ':')
        ax2.legend(loc='lower left', ncol=1, fontsize=12, frameon=False)
        plt.tight_layout()
        plt.savefig('Batch_result.png')   # 建议用 fig.savefig() 或 plt.savefig()，放在 show() 之前
        #plt.show()
    else:
        from scipy.stats import linregress

        fig, ax_tl = plt.subplots(1, 1, figsize=(5, 4), dpi=200)

        dir = "./Melt_Scenario2/oxi_standard/example_co2_15/"
        args    = readModelDataFile(dir + 'clean_args_array.dat')
        results = readModelDataFile(dir + 'clean_results.dat')

        frac, final_dia, peak_temp = [], [], []
        for i in range(len(args)):
            if results[i][1] < 1 and results[i][2] > 2000:
                frac.append(results[i][1])
                final_dia.append(results[i][0] * 2e6)
                peak_temp.append(results[i][2])

        final_dia = np.array(final_dia)
        frac      = np.array(frac)

        
        slope, intercept, r, p, se = linregress(final_dia, frac)
        r2     = r**2
        x_fit  = np.linspace(final_dia.min(), final_dia.max(), 200)
        y_fit  = slope * x_fit + intercept

    
        ax_tl.scatter(final_dia, frac,
                    color='steelblue', s=15, alpha=0.6, zorder=2,
                    label=r'CO$_2$ = 15 vol%, N$_2$ = 85 vol%')

        
        ax_tl.plot(x_fit, y_fit,
                color='tomato', lw=1.5, zorder=3,
                label=f'Linear fit ($R^2 = {r2:.3f}$)')

        ax_tl.set_xlim([0, 200])
        ax_tl.set_yticks([0.66, 0.7, 0.74,0.78,0.82,0.86, 0.90])
        ax_tl.set_xlabel('Final Diameter (µm)')
        ax_tl.set_ylabel('Metallic Fe Area Fraction')
        ax_tl.set_title('Final Diameter vs. Fe Fraction')
        ax_tl.legend(fontsize=9, frameon=False)
        ax_tl.grid(True, linestyle=':', alpha=0.3)

        fig.tight_layout()
        fig.savefig('batch_distri.pdf')
        plt.show()
        
