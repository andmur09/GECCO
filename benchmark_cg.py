import time
import pickle as pkl
from gecco import *
import LinearProgramParis as LP
def benchmark(PSTN, alpha):
    # Translates the PSTN to the standard form of a JCCP and stores the matrices in an instance of the JCCP class
    m, results = LP.solveLP(PSTN, PSTN.name)
    print("LP: ", m.objVal)
    m, results = gecco_algorithm(PSTN, tolog=True, logfile=PSTN.name + "_log_genetic")
    print("Genetic: ", m.objVal)
with open("pstns/problems/woodworking/p01", "rb") as f:
    problem = pkl.load(f)
    problem.plot()
benchmark(problem, 0.2)