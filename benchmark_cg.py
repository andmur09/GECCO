from column_generation_norm import *
import time
import pickle as pkl
import column_generation_norm
import column_generation_genetic
import LinearProgram2 as LP
def benchmark(PSTN, alpha):
    # Translates the PSTN to the standard form of a JCCP and stores the matrices in an instance of the JCCP class
    start = time.time() 
    #m, results = convex.solveJCCP(PSTN, alpha, 0.1, log=True, logfile=PSTN.name + "_log_standard")
    m, results = column_generation_genetic.genetic_solveJCCP(PSTN, alpha, 0.1, tolog=True, logfile=PSTN.name + "_log_genetic")
    print("Genetic: ", m.objVal)
    m, results = LP.solveLP(PSTN, PSTN.name, alpha)
    print("LP: ", m.objVal)
with open("pstns/problems/elevators/p05", "rb") as f:
    problem = pkl.load(f)
benchmark(problem, 0.2)