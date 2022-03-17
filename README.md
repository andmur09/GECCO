# JCC-PSTN

An implementation of the content from the paper "Joint Chance Constrained Probabilistic Temporal Networks
via Column Generation" submitted to SOCS 2022. Included is a python implementation of the column generation method for solving Strong Controllability of Joint Chance-Constrained Probabilistic Simple Temporal Networks, along with some test cases and results

## Quick Start
To run the experiments, Gurobi optimiser is required along with the gurobi python API GurobiPy. For details see:

https://www.gurobi.com/documentation/9.5/quickstart_linux/index.html

In order to solve a particular instance it is necessary to:

* Load the problem using `PSTN = pickle.load(f)`
* To Solve the JCC-PSTN problem via column generation 'convex.solveJCCP(PSTN, risk, gap, log, logfile, max_iterations, cg_tol)'
  - This should output a tuple (model, JCCP) where model is a Gurobi 'Model' object (for information on how to query parameters see https://www.gurobi.com/documentation/9.5/refman/py_model.html#pythonclass:Model) and 'JCCP' is an instance of 'JCCP' (for information on how to query parameters see 'JCCP_class.py')
* To Solve the problem using the LP do 'LP.solveLP(problem, name, risk)'.
  - This returns a tuple (model, result) where model is a Gurobi 'Model' object as above. Result is a dictionary '{"Runtime": float, "Objective": float, "Variable": value}'

In order to return the schedule from the Gurobi model, you can use 'getSchedule(PSTN, model)' and to get the relaxations use 'getRelaxations(PSTN, model)'.

## Examples

In the folder "pstns/problems" there are a number of problem cases generated from the IPC planning domains. PDDL domain, problem and plan files are stored in the folder "pstns/domains/tfd-benchmarks". 








