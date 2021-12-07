import numpy as np
from scipy.stats.mvn import mvnun as rectangular
import sys
import gurobipy as gp
from gurobipy import GRB
from math import sqrt, log

np.seterr(divide='raise')
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)
inf = 10000

def getStandardForm(PSTN):
    '''
    Description:    Makes matrices in the form Ax <= b and c^Tx representing the PSTN, which are used as input to the optimisation solver. For strong controllability Linear program.
    Input:          PSTN - Instance of PSTN to be solved
                    name - name to call model
                    pres - number of points for which to partition the function for probabilistic constraints (if pres = 50, then LHS of mode partitioned at 50 points, and RHS of mode partitioned at 50 points)
                    folder - folder to save Gurobi files to if log=True
                    log - if true writes Gurobi files to file
                    weight - weight to apply to relaxation cost terms in objective
    Output:         m - A Gurobi model containing all variables, constraints and objectives
    '''
    vars = PSTN.getProblemVariables()
    rvars = PSTN.getRandomVariables()
    cc = PSTN.getControllableConstraints()
    cu = PSTN.getUncontrollableConstraints()

    n = len(vars)
    m = 2 * len(cc)
    p = len(cu)
    r = len(rvars)

    c = np.zeros(n)
    A = np.zeros((m, n))
    b = np.zeros(m)
    T_l = np.zeros((p, n))
    q_l = np.zeros((p))
    T_u = np.zeros((p, n))
    q_u = np.zeros((p))
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
            rl_i = vars.index(cc[i].name + "_rl")
            A[ub, ru_i], c[ru_i] = -1, 1
            A[lb, rl_i], c[rl_i] = -1, 1
    
    # Gets matrices for bilateral joint chance constraint P(T_l + q_l <= Psi omega <= T_u + q_u) >= 1 - alpha
    for i in range(p):
        incoming = PSTN.incomingContingent(cu[i])
        if incoming["start"] != None:
            incoming = incoming["start"]
            start_i, end_i = vars.index(incoming.source.id), vars.index(cu[i].sink.id)
            T_u[i, start_i], T_u[i, end_i] = -1, 1
            T_l[i, start_i], T_l[i, end_i] = -1, 1
            q_u[i] = -cu[i].intervals["lb"]
            q_l[i] = -cu[i].intervals["ub"]
            if cu[i].hard == False:
                ru_i = vars.index(cu[i].name + "_ru")
                rl_i = vars.index(cu[i].name + "_rl")
                T_u[i, rl_i], c[ru_i] = -1, 1
                T_l[i, ru_i], c[rl_i] = -1, 1
            rvar_i = rvars.index("X" + "_" + incoming.source.id + "_" + incoming.sink.id)
            psi[i, rvar_i] = 1
            mu_X[rvar_i] = incoming.mu
            cov_X[rvar_i][rvar_i] = incoming.sigma**2
        elif incoming["end"] != None:
            incoming = incoming["end"]
            start_i, end_i = vars.index(cu[i].source.id), vars.index(incoming.source.id)
            T_u[i, start_i], T_u[i, end_i] = 1, -1
            T_l[i, start_i], T_l[i, end_i] = 1, -1
            q_u[i] = cu[i].intervals["ub"]
            q_l[i] = cu[i].intervals["lb"]
            if cu[i].hard == False:
                ru_i = vars.index(cu[i].name + "_ru")
                rl_i = vars.index(cu[i].name + "_rl")
                T_u[i, ru_i], c[ru_i] = 1, 1
                T_l[i, rl_i], c[rl_i] = 1, 1
            rvar_i = rvars.index("X" + "_" + incoming.source.id + "_" + incoming.sink.id)
            psi[i, rvar_i] = 1
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
    T_l, T_u = D @ T_l, D @ T_u
    q_l, q_u = D @ (q_l - mu_eta), D @ (q_u - mu_eta)    
    mu_xi = np.zeros((p))
    cov_xi = R
    return A, vars, b, c, T_l, T_u, q_l, q_u, mu_xi, cov_xi


def Initialise(A, vars, b, T_l, T_u, q_l, q_u):
    m = gp.Model("initialisation")
    x = m.addMVar(len(vars), vtype=GRB.CONTINUOUS, name="vars")
    t_l = m.addMVar(1, name="t_l")
    t_u = m.addMVar(1, name="t_u")
    p = T_l.shape[0]
    z_l = m.addMVar(p, name = "zl")
    z_u = m.addMVar(p, name = "zu")
    m.addConstr(A @ x <= b)
    m.addConstr(z_l == np.ones((p, 1)) @ t_l)
    m.addConstr(z_u == np.ones((p, 1)) @ t_u)
    m.addConstr(z_u <= T_u @ x + q_u)
    m.addConstr(z_l >= T_l @ x + q_l)
    m.addConstr(x[0] == 0)
    m.setObjective(t_u - t_l, GRB.MAXIMIZE)
    m.update()
    m.optimize()
    z_l = z_l.x
    z_u = z_u.x
    return (m, (z_l, z_u))

def masterProblem(A, vars, b, c, T_l, T_u, q_l, q_u, z, phi, pi):
    k = len(z)
    m = gp.model("iteration_" + str(k))
    x = m.addMVar(len(vars), name="vars")
    lam = m.addMVar(k, name="lambda")
    m.addConstr(A @ x <= b)
    m.addConstr(lam @ z <= T_u @ x + q_u)
    m.addConstr(lam @ z >= T_l @ x + q_l)
    m.addConstr(phi @ lam <= pi)
    m.addConstr(x[0] == 0)
    m.addConstr(lam.sum() == 1)
    m.setObjective(c @ x, GRB.MINIMIZE)
    m.update
    m.optimize()

def columnGeneration(v, u):
    pass

def JCCP(PSTN, alpha, epsilon):
    pi = -log(1-alpha)
    matrices = getStandardForm(PSTN)
    k = 0
    z = []
    A, vars, b, c, T_l, T_u, q_l, q_u, mu, cov = matrices[0], matrices[1], matrices[2], matrices[3], matrices[4], matrices[5], matrices[6], matrices[7], matrices[8], matrices[9]
    m, zk = Initialise(A, vars, b, T_l, T_u, q_l, q_u)
    if rectangular(zk[0], zk[1], mu, cov) >= 1 - alpha:
        z.append(zk)
        Obj_m, s, v, u = masterProblem(m, z)
        Obj_d, z_d = columnGeneration(v, u)
        while Obj_d > epsilon:
            k += 1
            zk = z_d
            z.append(zk)
            Obj_m, s, v, u = masterProblem(m, z)
            Obj_d, z_d = columnGeneration(v, u)
        return (s, Obj_m)
    else:
        raise ValueError("No solution could be found with current risk budget. Try increasing allowable risk")

def LPSolve(m, folder = None, log = False):
    '''
    Description:    Solves gurobi model and returns results
    Input:          m - unsolved gurobi model
                    output - boolean variable for additional information
    Output:         m - gurobi model after optimisation
    '''
    if log == True:
        m.write("{}/{}.mps".format(folder, m.ModelName))
    print("\n")
    m.optimize()
    print(m.status)
    if m.status == GRB.OPTIMAL:
        print('\n objective: ', m.objVal)
        print('\n Vars:')
        if log == True:
            m.write("{}/{}.sol".format(folder, m.ModelName))
        for v in m.getVars():
            print("Variable {}: ".format(v.varName) + str(v.x))
        return m

    elif m.status != GRB.OPTIMAL:
        if log == True:
            m.computeIIS()
            m.write("{}/{}.ilp".format(folder, m.ModelName))
        print('No solution')
        print("\n")
        return m
