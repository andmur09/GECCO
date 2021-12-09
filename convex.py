import numpy as np
from scipy.stats.mvn import mvnun as rectangular
from scipy.stats import multivariate_normal as norm
import sys
import gurobipy as gp
from gurobipy import GRB
from math import sqrt, log
from matplotlib import pyplot as plt

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

def Initialise(A, vars, b, T_l, T_u, q_l, q_u, box = 3):
    m = gp.Model("initialisation")
    x = m.addMVar(len(vars), vtype=GRB.CONTINUOUS, name="vars")
    t_l = m.addMVar(1, lb=-float(inf), name="t_l")
    t_u = m.addMVar(1, lb=-float(inf), name="t_u")
    p = T_l.shape[0]
    z_l = m.addMVar(p, lb=-float(inf), name = "zl")
    z_u = m.addMVar(p, lb=-float(inf), name = "zu")
    m.addConstr(A @ x <= b)
    m.addConstr(z_l == np.ones((p, 1)) @ t_l)
    m.addConstr(z_u == np.ones((p, 1)) @ t_u)
    m.addConstr(z_u <= T_u @ x + q_u)
    m.addConstr(z_l >= T_l @ x + q_l)
    m.addConstr(t_l >= -box)
    m.addConstr(t_u <= box)
    m.addConstr(x[0] == 0)
    m.setObjective(t_u - t_l, GRB.MAXIMIZE)
    m.update()
    m.optimize()
    if m.status == GRB.OPTIMAL:
        print('\n objective: ', m.objVal)
        print('\n Vars:')
        for v in m.getVars():
            print("Variable {}: ".format(v.varName) + str(v.x))
    z_l = z_l.x
    z_u = z_u.x
    return (m, (z_l, z_u))

def masterProblem(A, vars, b, c, T, q, z, phi, pi):
    k = np.shape(z)[1]
    p = len(q)
    m = gp.Model("iteration_" + str(k))
    x = m.addMVar(len(vars), name="vars")
    lam = m.addMVar(k, name="lambda")
    m.addConstr(A @ x <= b)
    for i in range(p):
        m.addConstr(T[i,:]@x + q[i] >= z[i, :]@lam, name="z{}".format(i))
    m.addConstr(phi @ lam <= pi, name="cc")
    m.addConstr(x[0] == 0)
    m.addConstr(lam.sum() == 1)
    m.setObjective(c @ x, GRB.MINIMIZE)
    m.update
    m.optimize()
    if m.status == GRB.OPTIMAL:
        print('\n objective: ', m.objVal)
        print('\n Vars:')
        for v in m.getVars():
            print("Variable {}: ".format(v.varName) + str(v.x))
    constraints = m.getConstrs()
    cnames = m.getAttr("ConstrName", constraints)
    u = []
    for i in range(len(cnames)):
        if cnames[i][0] == "z":
            u.append(constraints[i].getAttr("Pi"))
        elif cnames[i] == "cc":
            v = constraints[i].getAttr("Pi")
    return (m, np.c_[np.array(u)], v)            

def evaluateProb(z, mean, cov):
    size = int(np.shape(z)[0]/2)
    zl, zu = -1*z[:size], z[size:]
    F = rectangular(zl, zu, mean, cov)
    return F

def evaluateGrad(z, mean, cov):
    n = int(np.shape(mean)[0])
    size = int(np.shape(z)[0]/2)
    zl, zu = -1*z[:size], z[size:]
    dzl, dzu = [], []
    if n != np.shape(cov[0])[0] or n != np.shape(cov[1])[0] or n != np.shape(zl)[0]:
        raise AttributeError("Dimension of arrays are not compatible")
    for i in range(n):
        bar_mean_l = np.delete(mean, i)
        bar_mean_u = np.delete(mean, i)
        bar_cov = np.delete(np.delete(cov, i, 0), i, 1)
        bar_zl, bar_zu = np.delete(zl, i), np.delete(zu, i)

        bar_F_u = rectangular(bar_zl, bar_zu, bar_mean_u, bar_cov)[0]
        bar_F_l = rectangular(bar_zl, bar_zu, bar_mean_l, bar_cov)[0]
        xi = norm(mean[i], cov[i, i])
        fl = norm.pdf(zl[i])
        fu = norm.pdf(zu[i])
        dzu.append(fu * bar_F_u)
        dzl.append(-fl * bar_F_l)
    dzl = np.array(dzl)
    dzu = np.array(dzu)
    return np.vstack((np.c_[dzl], np.c_[dzu]))

def evaluateDual(z, v, u, mean, cov):
    F = evaluateProb(z, mean, cov)[0]
    phi = -log(F)
    return v * phi - np.transpose(u) @ z

def evaluateGradientDual(z, v, u, mean, cov):
    F = evaluateProb(z, mean, cov)[0]
    grad = evaluateGrad(z, mean, cov)
    return -v/F * grad - u

def backtracking(z, v, u, mean, cov, dual, grad, beta=0.8, alpha = 0.5):
    t = 1
    z_i = z - t * grad
    while evaluateDual(z-t*grad, v, u, mean, cov) > dual - alpha * t * grad.transpose() @ grad:
        t *= beta
    return t

def columnGeneration(z, v, u, mean, cov, iterations = 100):
    count = []
    f = []
    i = 1
    for i in range(iterations):
        # print("\nIteration {}".format(i))
        dual = evaluateDual(z, v, u, mean, cov)
        f.append(dual[0][0])
        # print("Function value = ", dual)
        grad = evaluateGradientDual(z, v, u , mean, cov)
        # print("Gradient Dualv= ", grad)
        t = backtracking(z, v, u, mean, cov, dual, grad)
        z = z - t * grad
        count.append(i)
        i += 1
    # print(f)
    # print(count)
    plt.plot(count, f)
    plt.savefig("grad.png", bbox_inches='tight')
    return z

def JCCP(PSTN, alpha, epsilon):
    pi = -log(1-alpha)
    matrices = getStandardForm(PSTN)
    k = 1
    m = []
    A, vars, b, c, T_l, T_u, q_l, q_u, mu, cov = matrices[0], matrices[1], matrices[2], matrices[3], matrices[4], matrices[5], matrices[6], matrices[7], matrices[8], matrices[9]
    res = Initialise(A, vars, b, T_l, T_u, q_l, q_u)
    m, zk = res[0], res[1]
    F = rectangular(zk[0], zk[1], mu, cov)[0]
    if F >= 1 - alpha:
        phi = np.c_[-log(F)]
        T = np.vstack((-T_l, T_u))
        q = np.vstack((-np.c_[q_l], np.c_[q_u]))
        z = np.vstack((-np.c_[zk[0]], np.c_[zk[1]]))
        m_master_k, u_k, v_k = masterProblem(A, vars, b, c, T, q, z, phi, pi)
        Obj_d, z_d = columnGeneration(np.c_[z[:, -1]], v_k, u_k, mu, cov)
        while Obj_d > epsilon:
            k += 1
            zk = z_d
            z.append(zk)
            Obj_m, s, v, u = masterProblem(m, z)
            Obj_d, z_d = columnGeneration(v, u)
        return (s, Obj_m)
    else:
        raise ValueError("No solution could be found with current risk budget. Try increasing allowable risk")
