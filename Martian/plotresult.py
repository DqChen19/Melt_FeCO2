import numpy as np
import matplotlib.pylab as plt
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
fname = "./batch_results/clean_results.dat"
results = readModelDataFile(fname)
particle_fractions = []
pure_ox_count = 0
pure_fe_count = 0

for j in range(len(results)):
    frac = results[j][1]
    rad = results[j][0]
    max_temp = results[j][2]
    
    particle_fractions.append(frac)

###plot the frac
particle_fractions = np.array(particle_fractions)

Modern_frac = [0.428877005, 0.484491979, 0.488770053, 0.418181818, 0.486631016,
               0.43315508, 0.443850267, 0.407486631, 0.407486631, 0.354010695,
               0.268449198, 0.178609626, 0.187165775, 0.28342246, 0.300534759,
               0.165775401, 0.206417112, 0.120855615, 0.328342246, 0.315508021,
               0.272727273, 0.165775401, 0.127272727, 0.103743316, 0.069518717,
               0.05026738, 0.018181818, 0.003208556, 0.17433155, 0.210695187,
               0.262032086, 0.165775401, 0.152941176,
               0.63231238, 0.14543235, 0.32836763, 0.38471399, 0.16137044,
               0.25564954, 0.12591051, 0.13312637, 0.44842156, 0.29907227,
               0.36, 0.3925098, 0.09347235, 0.11526444, 0.23701524, 0.12003224,
               0.0456571, 0.15923169, 0.04522839, 0.2401, 0.15340278,
               0.20971264, 0.17610013, 0.52892562, 0.15599181, 0.12351659]
print(len(Modern_frac))

Modern_frac = np.array(Modern_frac)
fig, ax = plt.subplots(figsize=(4.5,3.5), dpi=300)

# histogram as probability
weights = np.ones_like(particle_fractions) / len(particle_fractions)

bins=np.linspace(0,1,21)
mean = np.average(particle_fractions)
# location of error bar

ax.hist(
    particle_fractions,
    bins=bins,
    weights=np.ones_like(particle_fractions)/len(particle_fractions),
    alpha=0.7,
    color='orange',
    label='Modeled Mars ICSs'
)

ax.hist(
    Modern_frac,
    bins=bins,
    weights=np.ones_like(Modern_frac)/len(Modern_frac),
    alpha=0.7,
    color='royalblue',
    label='Modern Earth MET ICSs'
)

ax.set_xlabel('Metallic Fe area fraction')
ax.set_ylabel('Probability')

ax.set_xlim(0,1)
ax.set_xticks([0.0,0.2,0.4, 0.6, 0.8, 1.0])

ax.legend(frameon=False)

plt.tight_layout()
fig.savefig('compare.pdf')
plt.show()