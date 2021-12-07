from simulations import solve, getSchedule, getRelaxations
from gurobipy import GRB

def epsilonConstraint(pstn, folder, upper, epsilon, log=False):
    result = {}
    curr_budget = upper
    m = solve(pstn, folder, budget=curr_budget, log=log)
    if m.status == GRB.OPTIMAL:
        result['{:.2f}'.format(curr_budget)] = {"Schedule": getSchedule(pstn, m), "Relaxations": getRelaxations(pstn, m), "Risk": m.getVarByName("Risk").x, "Cost": m.getVarByName("Cost").x}
    curr_cost = m.getVarByName("Cost").x
    curr_budget -= epsilon
    m = solve(pstn, folder, budget=curr_budget, log=log)
    print(result)
    while m.status == GRB.OPTIMAL:
        new_cost = m.getVarByName("Cost").x
        if new_cost > curr_cost:
            m.update()
            result['{:.2f}'.format(curr_budget)] = {"Schedule": getSchedule(pstn, m), "Relaxations": getRelaxations(pstn, m), "Risk": m.getVarByName("Risk").x, "Cost": m.getVarByName("Cost").x}
        curr_budget -= epsilon
        m = solve(pstn, folder, budget = curr_budget, log=log)
    return {"PSTN": pstn, "Result": result}
