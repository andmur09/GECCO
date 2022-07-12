import time
import pickle as pkl
from gecco import *
from column_generation_LBFGSB import solve as LBFGSB
from column_generation_NM import solve as NM
import LinearProgramParis as LP
import monte_carlo as mc
import LinearProgramParis as LP
from simulations import getSchedule

def benchmark(PSTN):
    # Solves PSTN using a number of methods and compares results
    # m_ge, results_ge = gecco_algorithm(PSTN, tolog=True, logfile=PSTN.name + "_log_genetic")
    # print("Genetic: ", m_ge.objVal)
    # m_nm, results_nm = NM(PSTN, tolog=True, logfile = PSTN.name + "_log_nelder_mead")
    # print("Nelder Mead: ", m_nm.objVal)
    m_lb, results_lb = LBFGSB(PSTN, tolog=True, logfile = PSTN.name + "_log_l_bfgs_b")
    print("L-BFGS-B: ", m_lb.objVal)


    # # gets the schedule from the solution
    # schedule_nm = getSchedule(PSTN, m_nm)
    # print("Nelder Mead Schedule: ", schedule_nm)
    # schedule_lb = getSchedule(PSTN, m_lb)
    # print("L-BFGS-B Schedule: ", schedule_lb)
    # schedule_ge = getSchedule(PSTN, m_ge)
    # print("Genetic Schedule: ", schedule_ge)

    # # Simulates execution of the schedule to get Robustness
    # r_nm = mc.monte_carlo_success(PSTN, schedule_nm, 1)
    # r_lb = mc.monte_carlo_success(PSTN, schedule_lb, 1)
    # r_ge = mc.monte_carlo_success(PSTN, schedule_ge, 1)

    # print("\nNelder Mead:")
    # print("\tActual: ", r_nm)
    # print("\tEmpirical: ", results_nm.getCurrentProbability())

    # print("\nL-BFGS-B:")
    # print("\tActual: ", r_lb)
    # print("\tEmpirical: ", results_lb.getCurrentProbability())

    # print("\nGenetic:")
    # print("\tActual: ", r_ge)
    # print("\tEmpirical: ", results_ge.getCurrentProbability())

with open("pstns/problems/woodworking/p01_11_corr_00", "rb") as f:
    problem = pkl.load(f)
    problem.plot()
benchmark(problem)