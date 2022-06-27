#from os import getrandom
#from re import T
from curses.panel import top_panel
import numpy as np
#from scipy.stats.mvn import mvnun as rectangular
from scipy.stats import multivariate_normal as norm
from scipy import optimize
import sys
import gurobipy as gp
from gurobipy import GRB
from math import sqrt, log
import numpy as np
import additional_functions as fn
from JCCP_class import JCCP
import time
from timeout import timeout
import pygad
#from scipy.optimize import line_search

np.seterr(divide='raise')
np.set_printoptions(suppress=True)
np.set_printoptions(precision=8)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)
inf = 10000

def getStandardForm(PSTN):
    '''
    Description:    Makes matrices in the standard form of a Joint Chance Constrained Optimisation problem:

                    min     c^Tx

                    S.t.    A*vars <= b
                            P(xi <= T*vars + q ) >= 1 - alpha
                            xi = N(mu, cov)
    
    Input:          PSTN:   Instance of PSTN to be solved
    Output:         A:      m x n matrix of coefficients
                    vars:   n dimensional decision vector
                    b:      m dimensional vector of RHS values
                    c:      n dimensional vector of objective coefficients
                    T:      p x n matrix of coefficients
                    q:      p dimensional vector of RHS values
                    mu_xi:  p dimensional mean vector of xi
                    cov_xi: p x p dimensional correlation matrix of xi
    '''
    vars = PSTN.getProblemVariables()
    rvars = PSTN.getRandomVariables()
    cc = PSTN.getControllableConstraints()
    cu = PSTN.getUncontrollableConstraints()
    n = len(vars)
    m = 2 * len(cc)
    p = 2 * len(cu)
    r = len(rvars)

    c = np.zeros(n)
    A = np.zeros((m, n))
    b = np.zeros(m)
    T = np.zeros((p, n))
    q = np.zeros((p))
    mu_X = np.zeros((r))
    cov_X = np.zeros((r, r))
    psi = np.zeros((p, r))

    # Gets matrices for controllable constraints in form Ax <= b
    for i in range(len(cc)):
        ub = 2 * i
        lb = ub + 1
        start_i, end_i = vars.index(cc[i].source.id), vars.index(cc[i].sink.id)
        A[ub, start_i], A[ub, end_i], b[ub] = -1, 1, cc[i].intervals["ub"]
        A[lb, start_i], A[lb, end_i], b[lb] = 1, -1, -cc[i].intervals["lb"]
        if cc[i].hard == False:
            ru_i = vars.index(cc[i].name + "_ru")
            #rl_i = vars.index(cc[i].name + "_rl")
            A[ub, ru_i], c[ru_i] = -1, 1
            #A[lb, rl_i], c[rl_i] = -1, inf
    
    # Gets matrices for joint chance constraint P(Psi omega <= T * vars + q) >= 1 - alpha
    for i in range(len(cu)):
        ub = 2 * i
        lb = ub + 1
        incoming = PSTN.incomingContingent(cu[i])
        if incoming["start"] != None:
            incoming = incoming["start"]
            start_i, end_i = vars.index(incoming.source.id), vars.index(cu[i].sink.id)
            T[ub, start_i], T[ub, end_i] = 1, -1
            T[lb, start_i], T[lb, end_i] = -1, 1
            q[ub] = cu[i].intervals["ub"]
            q[lb] = -cu[i].intervals["lb"]
            if cu[i].hard == False:
                ru_i = vars.index(cu[i].name + "_ru")
                #rl_i = vars.index(cu[i].name + "_rl")
                T[ub, ru_i], c[ru_i] = 1, 1
                #T[lb, rl_i], c[rl_i] = 1, inf
            rvar_i = rvars.index("X" + "_" + incoming.source.id + "_" + incoming.sink.id)
            psi[ub, rvar_i] = -1
            psi[lb, rvar_i] = 1
            mu_X[rvar_i] = incoming.mu
            cov_X[rvar_i][rvar_i] = incoming.sigma**2
        elif incoming["end"] != None:
            incoming = incoming["end"]
            start_i, end_i = vars.index(cu[i].source.id), vars.index(incoming.source.id)
            T[ub, start_i], T[ub, end_i] = 1, -1
            T[lb, start_i], T[lb, end_i] = -1, 1
            q[ub] = cu[i].intervals["ub"]
            q[lb] = -cu[i].intervals["lb"]
            if cu[i].hard == False:
                ru_i = vars.index(cu[i].name + "_ru")
                #rl_i = vars.index(cu[i].name + "_rl")
                T[ub, ru_i], c[ru_i] = 1, 1
                #T[lb, rl_i], c[rl_i] = 1, inf
            rvar_i = rvars.index("X" + "_" + incoming.source.id + "_" + incoming.sink.id)
            psi[ub, rvar_i] = 1
            psi[lb, rvar_i] = -1
            mu_X[rvar_i] = incoming.mu
            cov_X[rvar_i][rvar_i] = incoming.sigma**2
        else:
            raise AttributeError("Not an uncontrollable constraint since no incoming pstc")

    # Performs transformation of X into eta where eta = psi X such that eta is a p dimensional random variable
    mu_eta = psi @ mu_X
    cov_eta = psi @ cov_X @ psi.transpose()

    # Translates random vector eta into standard form xi = N(0, R) where R = D.eta.D^T
    D = np.zeros((p, p))
    for i in range(p):
        D[i, i] = 1/sqrt(cov_eta[i, i])
    R = D @ cov_eta @ D.transpose()
    T = D @ T
    q = D @ (q - mu_eta)
    mu_xi = np.zeros((p))
    cov_xi = R
    return A, vars, b, c, T, q, mu_xi, cov_xi
    
def Initialise(JCCP, box = 6):
    '''
    Description:    Finds an initial feasible point such that the joint chance constraint is satisfied. Solves the following problem:

                    max     t

                    S.t.    A*vars <= b
                            z <= T*vars + q
                            z = 1 t
                            t <= box

                    And checks to see whether the point z satisfies the chance constraint P(xi <= z) >= 1 - alpha.
    
    Input:          JCCP:   Instance of JCCP class
                    box:    no of standard deviations outside which should be neglected
    Output:         m:      An instance of the Gurobi model class
    '''
    # Sets up and solves Gurobi opimisation problem
    m = gp.Model("initialisation")
    x = m.addMVar(len(JCCP.vars), vtype=GRB.CONTINUOUS, name="vars")
    t = m.addMVar(1, name="t_l")
    p = JCCP.T.shape[0]
    z = m.addMVar(p, lb=-float(inf), name = "zl")
    m.addConstr(JCCP.A @ x <= JCCP.b)
    m.addConstr(z == np.ones((p, 1)) @ t)
    m.addConstr(z <= JCCP.T @ x + JCCP.q)
    m.addConstr(t <= box)
    m.addConstr(x[JCCP.start_i] == 0)
    m.setObjective(t, GRB.MAXIMIZE)
    m.update()
    m.write("convex.lp")
    m.optimize()

    # Checks to see whether an optimal solution is found and if so it prints the solution and objective value
    if m.status == GRB.OPTIMAL:
        print('\n objective: ', m.objVal)
        print('\n Vars:')
        for v in m.getVars():
            print("Variable {}: ".format(v.varName) + str(v.x))
    else:
        m.computeIIS()
        m.write("convex.ilp")

    z_ = np.array(z.x)

    # Checks to see whether solution to z satisfies the chance constraint
    F0 = fn.prob(z_, JCCP.mean, JCCP.cov)
    phi = []
    # If chance constraint is satisfies adds p approximation points z^i = (z_1 = t,..,z_i = 0,..,z_p = t) for i = 1,2,..,p
    if F0 >= 1 - JCCP.alpha:
        phi.append(-log(F0))
        z = np.c_[z_]
        for i in range(len(z_)):
            znew = np.copy(z_)
            znew[i] = 0
            z = np.hstack((z, np.c_[znew]))
            Fnew = fn.prob(znew, JCCP.mean, JCCP.cov)
            phi.append(-log(Fnew))
    else:
        return None
    # Initialises the matrix z and vector phi within the instance of JCCP
    JCCP.setZ(z)
    JCCP.setPhi(np.array(phi))
    return m

def masterProblem(JCCP):
    '''
    Description:    Solves the restricted master problem:

                    min.    c^Tx

                    S.t.    A*vars <= b
                            T*vars + q >= sum_{i=0}^k{lambda^i z^i}
                            sum_{i=0}^k{lambda^i} = 1
                            sum_{i=0}^k{phi^i * lambda^i} <= pi
                            lambda^i >= 0

                    And returns a Gurobi model instance containing solution and optimal objective for current iteration.
    
    Input:          JCCP:   Instance of JCCP class
    Output:         m:      An instance of the Gurobi model class
                    zsol:   Result sum_{i=0}^k{lambda^i z^i}
    '''
    # Sets up and solves the restricted master problem
    k = np.shape(JCCP.z)[1]
    p = len(JCCP.q)
    m = gp.Model("iteration_" + str(k))
    x = m.addMVar(len(JCCP.vars), name=JCCP.vars)
    lam = m.addMVar(k, name="lambda")
    phi = m.addMVar(1, name = "phi")
    m.addConstr(JCCP.A @ x <= JCCP.b, name="cont")
    for i in range(p):
        m.addConstr(JCCP.z[i, :]@lam <= JCCP.T[i,:]@x + JCCP.q[i], name="z{}".format(i))
    m.addConstr(lam @ JCCP.phi <= JCCP.getPi(), name="cc")
    m.addConstr(x[JCCP.start_i] == 0, name="x0")
    m.addConstr(lam.sum() == 1, name="sum_lam")
    m.addConstr(lam @ JCCP.phi == phi, 'phi')
    m.setObjective(JCCP.c @ x, GRB.MINIMIZE)
    m.update()
    m.write("convex.lp")
    m.optimize()

    # Checks to see whether an optimal solution is found and if so it prints the solution and objective value
    if m.status == GRB.OPTIMAL:
        print('\n objective: ', m.objVal)
        print('\n Vars:')
        for v in m.getVars():
            print("Variable {}: ".format(v.varName) + str(v.x))
        m.write("convex.sol")

    # Queries Gurobi to get values of dual variables and cbasis
    constraints = m.getConstrs()
    cnames = m.getAttr("ConstrName", constraints)
    u, mu, cb = [], [], []
    for i in range(len(cnames)):
        if cnames[i][0] == "z":
            u.append(constraints[i].getAttr("Pi"))
            cb.append(constraints[i].getAttr("CBasis"))
        elif cnames[i] == "cc":
            v = constraints[i].getAttr("Pi")
        elif cnames[i] == "sum_lam":
            nu = constraints[i].getAttr("Pi")
        elif cnames[i][0:4] == "cont":
            mu.append(constraints[i].getAttr("Pi"))

    mu = np.c_[np.array(mu)]
    u = np.c_[np.array(u)]

    # Sets the dual values and cbasis within the instance of JCCP to the optimal value for the current iteration
    JCCP.setDuals({"u": u, "v": v, "nu": nu, "mu": mu})
    JCCP.setCbasis(np.array(cb))
    #print("New Dual Variables added: ", JCCP.duals)

    # Gets values for variables lambda and evaluates current value of sum_{i=0}^k{lambda^i z^i} and sum(i=0)^k{lambda^i phi^i}
    lam_sol = np.array(lam.x)
    z_sol = np.array(sum([lam_sol[i]*JCCP.z[:, i] for i in range(np.shape(JCCP.z)[1])]))
    return (m, np.c_[z_sol])

def columnGeneration(z, JCCP, tol, maxiter=15000, maxls=20):
    '''
    Description:    Solves the column generaion problem (below) via SciPy optimize:

                    min_z.  -u^Tz - v*phi(z) - nu    

                    Every time we evaluate phi(z) we save the result to new_cols and new_phis return all new
    
    Input:          JCCP:       Instance of JCCP class
    Output:         columns:    Matrix of new columns to be added
                    values:     Vector of phi values to be added
                    f:          Final function value (reduced cost)
                    status:     Boolean stating whether the optimisation was successful
    '''
    global new_cols
    new_cols = []

    global new_phis
    new_phis = []
 
    duals = JCCP.getDuals()
    u, v, nu = fn.flatten(duals["u"]), duals["v"], duals["nu"]
    mean, cov = JCCP.mean, JCCP.cov
    cb = JCCP.cbasis
    z = fn.flatten(z)
    start = time.time()

    def dualf(z):
        phi = -log(norm(mean, cov, allow_singular=True).cdf(z))
        f = -np.dot(u, z) - v * phi - nu
        if f <= 0:
            # Keeps track of new columns in global variable and adds all points that have positive
            # reduced cost. This allows us to add multiple points at a time
            global new_cols
            new_cols.append(z)
            global new_phis
            new_phis.append(phi)
        return f

    def gradf(z):
        # Nested function to calculate gradients at particular points
        return fn.flatten(v/norm(mean, cov, allow_singular=True).cdf(z) * fn.grad(np.c_[z], cb, mean, cov)) - u
    
    # Adds bounds to prevent variables being non-negative
    bounds = []
    for i in range(len(z)):
        bound = (0.00001, 6)
        bounds.append(bound)

    res = optimize.minimize(dualf, z, jac = gradf, method = "L-BFGS-B", bounds=bounds, options={'ftol': tol, 'maxiter': maxiter, 'maxls': maxls})
    end = time.time()
    print("Time taken: ", end - start)
    print("\n", res)
    z = res.x
    f = res.fun
    status = res.success

    columns = []
    for z in new_cols:
        columns.append(np.c_[z])
    values = new_phis[:]

    return columns, values, f, status

def genetic_column_generation(JCCP):
    global new_cols_g
    new_cols_g = []

    global new_phis_g
    new_phis_g = []

    duals = JCCP.getDuals()
    u, v, nu = fn.flatten(duals["u"]), duals["v"], duals["nu"]
    mean, cov = JCCP.mean, JCCP.cov
    

    def genetic_dualf(z, solution_idx):
        try:
            phi = -log(norm(mean, cov, allow_singular=True).cdf(z))
            f = np.dot(u, z) + v * phi + nu
        except:
            f = 0
        if f > 0:
           #  Keeps track of new columns in global variable and adds all points that have positive
           # reduced cost. This allows us to add multiple points at a time
           global new_cols_g
           new_cols_g.append(z)
           global new_phis_g
           new_phis_g.append(phi)
        return f
    
    ga = pygad.GA(num_generations=10,
                    num_parents_mating=2,
                    fitness_func=genetic_dualf,
                    sol_per_pop=8,
                    num_genes=len(u),
                    init_range_low=0,
                    init_range_high=6,
                    save_best_solutions=True
    )
    ga.run()
    obj = -ga.best_solutions_fitness[-1]
    print("Reduced Cost : ", )
    print(ga.best_solutions_fitness)
    print(ga.best_solutions)
    columns = []
    for z in new_cols_g:
        columns.append(np.c_[z])
    values = new_phis_g[:]
    return columns, values, obj
    
def solveJCCP(PSTN, alpha, epsilon, log=False, logfile = None, max_iterations = 100, cg_tol = 0.05):
    '''
    Description:    Solves the problem of a joint chance constrained PSTN strong controllability via primal-dual column
                    generation method.
    
    Input:          PSTN:           Instance of PSTN class
                    alpha:          Allowable tolerance on risk:
                                    e.g. P(success) >= 1 - alpha
                    epsilon:        An allowable upper bound on the distance between the current solution and the global optimum
                                    e.g. (UB - LB)/LB <= epsilon    
                    log:            Boolean, whether or not to print to log file
                    logfile:        File to save log to
                    max_iteraions:  Option to set maxmimum number of iterations
                    cg_tol:         Tolerance to use with Column Generation optimisation (see: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
    Output:         m:              An instance of the Gurobi model class which solves the joint chance constrained PSTN
                    problem:        An instance of the JCCP class containing problem results
    '''
    n_iterations = 0
    LB = 0.0001

    if log == True:
        saved_stdout = sys.stdout
        sys.stdout = open("logs/{}.txt".format(logfile), "w+")
    
    # Translates the PSTN to the standard form of a JCCP and stores the matrices in an instance of the JCCP class
    start = time.time()
    matrices = getStandardForm(PSTN)
    A, vars, b, c, T, q, mu, cov = matrices[0], matrices[1], matrices[2], matrices[3], matrices[4], matrices[5], matrices[6], matrices[7]
    problem = JCCP(A, vars, b, c, T, q, mu, cov, alpha)
    problem.start_i = problem.vars.index(PSTN.getStartTimepointName())
    
    # Initialises the problem with k approximation points
    m = Initialise(problem)
    if m == None:
        print("No solution possible with current risk bound")
        if log == True:
            sys.stdout.close()
            sys.stdout = saved_stdout
            return None
    k = len(problem.phi)

    # Solves the master problem
    print("\nSolving master problem with {} approximation points".format(k))
    m, z_m = masterProblem(problem)
    problem.add_master_time(time.time() - start, m.objVal)
    problem.addSolution(m)
    print("Current objective is: ", m.objVal)
    UB = m.objVal

    # Solves the column generation problem
    print("\nSolving Column Generation")
    z_d, vals, f, status = columnGeneration(z_m, problem, cg_tol)
    rho = -f

    # If column generation terminates succesfully, updates the lower bound
    if status == True:
        print("Calculating new lower bound: ", "obj = ", m.objVal, "rho = ", rho, "rho(1 + cg_tol) = ", rho*(1 + cg_tol))
        LB = m.objVal - rho*(1 + cg_tol)
        problem.add_convergence_time(time.time() - start, (UB - LB)/LB)
        print("lower bound is: ", LB)
    # Adds column and Repeats process until acceptable tolerance on optimalty gap is attained
    #while abs((UB - LB)/LB) > epsilon and n_iterations <= max_iterations and rho >= 0:
    while n_iterations <= max_iterations and rho >= 0:
        n_iterations += 1
        k += 1
        
        # Adds new points from column generation procedure
        for i in range(len(vals)):
            problem.addColumn(z_d[i], vals[i])

        print("\nSolving master problem with {} approximation points".format(k))
        m, z_m = masterProblem(problem)
        problem.add_master_time(time.time() - start, m.objVal)
        problem.addSolution(m)
        print("Current objective is: ", m.objVal)
        UB_temp = m.objVal
        
        # Checks to see if desired optimality gap is attained and if so it breaks and
        # terminates the algorithm.
        UB = UB_temp
        if abs((UB - LB)/LB) <= epsilon:
            break

        print("\nSolving Column Generation")
        z_d, vals, f, status = columnGeneration(z_m, problem, cg_tol)
        rho = -f

        if status == True:
            print("Calculating new lower bound: ", "obj = ", m.objVal, "rho = ", rho, "rho(1 + cg_tol) = ", rho*(1 + cg_tol))
            LB_k = m.objVal - rho*(1 + cg_tol)
            LB = max(LB, LB_k)
            problem.add_convergence_time(time.time() - start, (UB - LB)/LB)
            print("lower bound is: ", LB)
            
        print("LB = ", LB, "UB = ", UB)

    end = time.time()
    solution_time = end - start
    if n_iterations <= max_iterations:
        problem.setSolved(True)

    problem.setSolutionTime(solution_time)
    print("\nFinal solution found: ")
    print("Solution time: ", solution_time)
    print("Optimality gap is: ", (UB - LB)/LB*100)
    print("Final Probability is: ", problem.getCurrentProbability())
    print('objective: ', m.objVal)
    print('Vars:')
    for v in m.getVars():
        print("Variable {}: ".format(v.varName) + str(v.x))
    if log == True:
        sys.stdout.close()
        sys.stdout = saved_stdout
    return m, problem

def genetic_solveJCCP(PSTN, alpha, epsilon, log=False, logfile = None, max_iterations = 100):
    '''
    Description:    Solves the problem of a joint chance constrained PSTN strong controllability via primal-dual column
                    generation method.
    
    Input:          PSTN:           Instance of PSTN class
                    alpha:          Allowable tolerance on risk:
                                    e.g. P(success) >= 1 - alpha
                    epsilon:        An allowable upper bound on the distance between the current solution and the global optimum
                                    e.g. (UB - LB)/LB <= epsilon    
                    log:            Boolean, whether or not to print to log file
                    logfile:        File to save log to
                    max_iteraions:  Option to set maxmimum number of iterations
                    cg_tol:         Tolerance to use with Column Generation optimisation (see: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
    Output:         m:              An instance of the Gurobi model class which solves the joint chance constrained PSTN
                    problem:        An instance of the JCCP class containing problem results
    '''
    n_iterations = 100
    if log == True:
        saved_stdout = sys.stdout
        sys.stdout = open("logs/{}.txt".format(logfile), "w+")
    
    # Translates the PSTN to the standard form of a JCCP and stores the matrices in an instance of the JCCP class
    start = time.time()
    matrices = getStandardForm(PSTN)
    A, vars, b, c, T, q, mu, cov = matrices[0], matrices[1], matrices[2], matrices[3], matrices[4], matrices[5], matrices[6], matrices[7]
    problem = JCCP(A, vars, b, c, T, q, mu, cov, alpha)
    problem.start_i = problem.vars.index(PSTN.getStartTimepointName())
    
    # Initialises the problem with k approximation points
    m = Initialise(problem)
    if m == None:
        print("No solution possible with current risk bound")
        if log == True:
            sys.stdout.close()
            sys.stdout = saved_stdout
            return None
    k = len(problem.phi)

    # Solves the master problem
    print("\nSolving master problem with {} approximation points".format(k))
    m, z_m = masterProblem(problem)
    problem.add_master_time(time.time() - start, m.objVal)
    problem.addSolution(m)
    print("Current objective is: ", m.objVal)
    UB = m.objVal

    # Solves the column generation problem
    print("\nSolving Column Generation")
    z_d, vals, cg_obj = genetic_column_generation(problem)
    # Adds column and Repeats process until acceptable tolerance on optimalty gap is attained
    while n_iterations <= max_iterations and cg_obj > 0:
        k += 1
        # Adds new points from column generation procedure
        for i in range(len(vals)):
            problem.addColumn(z_d[i], vals[i])
        
        print("\nSolving master problem with {} approximation points".format(k))
        m, z_m = masterProblem(problem)
        problem.add_master_time(time.time() - start, m.objVal)
        problem.addSolution(m)
        print("Current objective is: ", m.objVal)
        #UB_temp = m.objVal

        print("\nSolving Column Generation")
        z_d, vals, cg_obj = genetic_column_generation(problem)

    end = time.time()
    solution_time = end - start
    if n_iterations <= max_iterations:
        problem.setSolved(True)

    problem.setSolutionTime(solution_time)
    print("\nFinal solution found: ")
    print("Solution time: ", solution_time)
    print("Final Probability is: ", problem.getCurrentProbability())
    print('objective: ', m.objVal)
    print('Vars:')
    for v in m.getVars():
        print("Variable {}: ".format(v.varName) + str(v.x))
    if log == True:
        sys.stdout.close()
        sys.stdout = saved_stdout
    return m, problem


