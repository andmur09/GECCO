from os import getrandom
from re import T
import numpy as np
from scipy.stats.mvn import mvnun as rectangular
from scipy.stats import multivariate_normal as norm
import sys
import gurobipy as gp
from gurobipy import GRB
from math import sqrt, log
from matplotlib import pyplot as plt
import compute_probabilities as prob
import numpy as np
import additional_functions as fn

np.seterr(divide='raise')
np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)
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
            rl_i = vars.index(cc[i].name + "_rl")
            A[ub, ru_i], c[ru_i] = -1, 1
            A[lb, rl_i], c[rl_i] = -1, 1
    
    # Gets matrices for bilateral joint chance constraint P(T_l + q_l <= Psi omega <= T_u + q_u) >= 1 - alpha
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
                rl_i = vars.index(cu[i].name + "_rl")
                T[ub, ru_i], c[ru_i] = 1, 1
                T[lb, rl_i], c[rl_i] = 1, 1
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
                rl_i = vars.index(cu[i].name + "_rl")
                T[ub, ru_i], c[ru_i] = 1, 1
                T[lb, rl_i], c[rl_i] = 1, 1
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

def Initialise(A, vars, b, T, q, alpha, mu, cov, box = 3, epsilon = 0.1):
    m = gp.Model("initialisation")
    x = m.addMVar(len(vars), vtype=GRB.CONTINUOUS, name="vars")
    t = m.addMVar(1, lb=-float(inf), name="t_l")
    p = T.shape[0]
    z = m.addMVar(p, lb=-float(inf), name = "zl")
    m.addConstr(A @ x <= b)
    m.addConstr(z == np.ones((p, 1)) @ t)
    m.addConstr(z <= T @ x + q)
    m.addConstr(t <= box)
    m.addConstr(x[0] == 0)
    m.setObjective(t, GRB.MAXIMIZE)
    m.update()
    m.optimize()
    if m.status == GRB.OPTIMAL:
        print('\n objective: ', m.objVal)
        print('\n Vars:')
        for v in m.getVars():
            print("Variable {}: ".format(v.varName) + str(v.x)),
    z_ = np.array(z.x)
    F0 = prob.pmvnorm(z_, mu, cov)
    phi = []
    if F0 >= 1 - alpha:
        phi.append(-log(F0))
        z = np.c_[z_]
        for i in range(len(z_)):
            znew = np.copy(z_)
            znew[i] = 0
            z = np.hstack((z, np.c_[znew]))
            Fnew = prob.pmvnorm(znew, mu, cov)
            phi.append(-log(Fnew))
    else:
        raise ValueError("No solution could be found with current risk budget. Try increasing allowable risk")
    return (m, z, np.array(phi))

def masterProblem(A, vars, b, c, T, q, z, phi, pi, epsilon = 0.0001):
    k = np.shape(z)[1]
    p = len(q)
    m = gp.Model("iteration_" + str(k))
    x = m.addMVar(len(vars), name="vars")
    lam = m.addMVar(k, name="lambda")
    m.addConstr(A @ x <= b, name="cont")
    for i in range(p):
        m.addConstr(z[i, :]@lam <= T[i,:]@x + q[i], name="z{}".format(i))
    m.addConstr(lam @ phi <= pi, name="cc")
    m.addConstr(x[0] == 0, name="x0")
    m.addConstr(lam.sum() == 1, name="sum_lam")
    m.setObjective(c @ x, GRB.MINIMIZE)
    m.update
    m.write("convex.lp")
    m.optimize()

    if m.status == GRB.OPTIMAL:
        print('\n objective: ', m.objVal)
        print('\n Vars:')
        for v in m.getVars():
            print("Variable {}: ".format(v.varName) + str(v.x))
    m.write("convex.sol")

    # Queries Gurobi to get values of dual variables
    constraints = m.getConstrs()
    cnames = m.getAttr("ConstrName", constraints)
    cbasis = m.getAttr("CBasis", constraints)

    u = []
    mu = []
    for i in range(len(cnames)):
        if cnames[i][0] == "z":
            u.append(constraints[i].getAttr("Pi"))
        elif cnames[i] == "cc":
            v = constraints[i].getAttr("Pi")
        elif cnames[i] == "sum_lam":
            nu = constraints[i].getAttr("Pi")
        elif cnames[i][0:4] == "cont":
            mu.append(constraints[i].getAttr("Pi"))
    mu = np.array(mu)
    u = np.array(u)

    # Gets values for variables lambda and x
    lam_sol = np.array(lam.x)
    x_sol = np.array(x.x)

    # Gets values for sum_i^k (phi_i lambda_i) and sum_i^k (lambda_i z_i)
    z_sol = np.array(sum([lam_sol[i]*z[:, i] for i in range(np.shape(z)[1])]))
    phi_sol = np.dot(lam_sol, phi)
    print("Phi_sol = ", phi_sol, "Pi", pi)
    print("v = ", v)
    print("u = ", u)
    print("nu = ", nu)
    print("mu = ", mu)
    return (m, np.c_[z_sol], np.c_[np.array(u)], v, nu)

def backtracking(z, v, u, mean, cov, dual, grad, beta=0.8, alpha = 0.5):
    t = 1
    try:
        fn.dual(u, v, z-t*grad, mean, cov)
    except ValueError:
        t *= beta
    while fn.dual(u, v, z-t*grad, mean, cov) > dual - alpha * t * grad.transpose() @ grad:
        t *= beta
    return t

def columnGeneration(z, v, u, mean, cov, iterations = 1):
    grad = fn.gradDual(u, v, z, mean, cov)
    for i in range(iterations):
        dual = fn.dual(u, v, z, mean, cov)
        grad = fn.gradDual(u, v, z, mean, cov)
        t = backtracking(z, v, u, mean, cov, dual, grad)
        z = z - t * grad
        
        print("z = ", z)
        print("Function value = ", fn.dual(u, v, z, mean, cov))
    # print(f)
    # print(count)
   # plt.plot(count, f)
    #plt.savefig("grad.png", bbox_inches='tight')
    return z

def JCCP(PSTN, alpha, epsilon):
    #sys.stdout = open("log.txt", "w")
    ks = []
    objs = []
    pi = -log(1-alpha)
    matrices = getStandardForm(PSTN)
    m = []
    A, vars, b, c, T, q, mu, cov = matrices[0], matrices[1], matrices[2], matrices[3], matrices[4], matrices[5], matrices[6], matrices[7]
    res = Initialise(A, vars, b, T, q, alpha, mu, cov)
    m, zk, phi = res[0], res[1], res[2]
    k = len(phi)
    print("\nSolving master problem with {} approximation points".format(k))
    m_master_k, z_m, u_k, v_k, nu_k = masterProblem(A, vars, b, c, T, q, zk, phi, pi)
    ks.append(k)
    objs.append(m_master_k.objVal)
    print("\nCurrent z points are: ", zk)
    print("Current objective is: ", m_master_k.objVal)
    print("\nSolving Column Generation")
    z_d = columnGeneration(z_m, v_k, u_k, mu, cov)
    print("\nNew approximation point found: ", z_d)
    print("Reduced cost is: ", fn.reducedCost(u_k, v_k, nu_k, z_d, mu, cov))
    while fn.reducedCost(u_k, v_k, nu_k, z_d, mu, cov) > epsilon:
        k += 1
        zk = np.hstack((zk, z_d))
        phi_k = -log(prob.pmvnorm(z_d, mu, cov))
        phi = np.append(phi, phi_k)
        print("\nSolving master problem with {} approximation points".format(k))
        m_master_k, z_m, u_k, v_k, nu_k = masterProblem(A, vars, b, c, T, q, zk, phi, pi)
        ks.append(k)
        objs.append(m_master_k.objVal)
        print("\nCurrent z points are: ", zk)
        print("Current objective is: ", m_master_k.objVal)
        print("\nSolving Column Generation")
        z_d = columnGeneration(z_m, v_k, u_k, mu, cov)
        print("\nNew approximation point found: ", z_d)
        print("Reduced cost is: ", fn.reducedCost(u_k, v_k, nu_k, z_d, mu, cov))
    #sys.stdout.close()
    #plt.plot(ks, objs)
    #plt.savefig('objectrive.png')
    return None
