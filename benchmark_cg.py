from convex import *
import time

def benchmark(PSTN, alpha):
    # Translates the PSTN to the standard form of a JCCP and stores the matrices in an instance of the JCCP class
    start = time.time()
    matrices = getStandardForm(PSTN)
    A, vars, b, c, T, q, mu, cov = matrices[0], matrices[1], matrices[2], matrices[3], matrices[4], matrices[5], matrices[6], matrices[7]
    problem = JCCP(A, vars, b, c, T, q, mu, cov, alpha)
    problem.start_i = problem.vars.index(PSTN.getStartTimepointName())
    
    # Initialises the problem with k approximation points
    m = Initialise(problem)
    k = len(problem.phi)

    # Solves the master problem
    print("\nSolving master problem with {} approximation points".format(k))
    m, z_m = masterProblem(problem)
    problem.add_master_time(time.time() - start, m.objVal)
    problem.addSolution(m)
    print("Current objective is: ", m.objVal)

    # Solves the column generation problem
    print("\nSolving Column Generation")
    start = time.time()
    z_d, vals, f, status = columnGeneration(z_m, problem, 0.05)
    print("Obj", f)
    print("Runtime", time.time() - start)

    print("\nSolving Genetic Column Generation")
    start = time.time()
    z_d, vals, cg_obj = genetic_column_generation(problem)
    print("Obj", cg_obj)
    print("Runtime", time.time() - start)