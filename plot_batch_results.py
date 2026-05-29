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
        pure_ox_frac[i] = 1 if total_ox == 0 else pure_ox_count / total_ox
        
        if len(particle_fractions) > 0:
            means[i] = np.mean(particle_fractions)
            std = np.std(particle_fractions)
        else:
            means[i] = 0
            std = 0

        std_tops[i] = means[i] + 2*std
        std_bots[i] = means[i] - 2*std
        co2_percents[i] = val
    return co2_percents,means, std_tops, std_bots, pure_ox_frac

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
    f_size = 14
    ax.tick_params(labelsize=f_size)
    ax.set_xlabel(r"Atmospheric CO$_2$ [Vol%]", fontsize=f_size)
    ax.set_ylabel("Fe Fractional Area", fontsize=f_size)

    ax.plot(co2_percents, means, '-', color=package['color'])
    ax.fill_between(co2_percents, std_bots, std_tops, color=package['color'], alpha=0.35, label=package['label'])

tomkins_e = 0.726
tomkins_f = 0.024

if __name__ == '__main__':

    fig, ax = plt.subplots(figsize=(6, 4), dpi=220)
    dir_1 = np.arange(5,91,10)
    package1 = {'color':'blue','label':'Kinetic'}
    plot_co2_data(ax, dir="./Melt_Scenario2/oxi_standard", dir_list= dir_1, package=package1)
    package2 = {'color':'orange', 'label':'Add all'}
    dir_2 = np.arange(2,92,5)
    plot_co2_data(ax, dir="./Scenario1/add_standard",dir_list=dir_2, package=package2)


    plt.legend(loc='lower left',
            ncol=1, fontsize=12, frameon=False)
    ax.set_xlim(5,85)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(5, 85, 10))
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()
    fig.savefig('Batch_result.pdf')