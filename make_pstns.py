from  parse_json import JSONtoPSTN
import pickle as pkl
import os
import numpy as np
from additional_functions import generate_random_correlation

directories = ["car_data", "rover_data"]
for directory in directories:
    for file in os.listdir("pstns/jsons/" + directory):
        if "json" in file:
            name = file[:-5]
            # Parses JSON and returns a PSTN
            network = JSONtoPSTN("pstns/jsons/{}/{}".format(directory, file), name)
            # Duplicates the PSTN adding random correlation, making 10 corr-PSTNs for each PSTN
            rvs = network.getContingents()
            for k in np.linspace(0.1, 1, 10):
                correlation = generate_random_correlation(len(rvs), k)
                eta = str(round(k, 1)).split(".")
                eta = eta[0] + eta[1]
                new_network = network.makeCopyWithCorrelation(network.name + "_" + eta, correlation)
                with open("pstns/problems/{}/{}".format(directory, new_network.name), "wb") as f:
                    pkl.dump(new_network, f)
