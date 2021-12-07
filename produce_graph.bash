#!/bin/bash

# arguments expected:
# - path from pwd to the domain and problem files
# - name of the domain file
# - name of the problem file

path=$(pwd)/$1
domain=$path/$2
problem=$path/$3
problem_name=$3

# the launch file should be in the working directory
roslaunch generate_dot.launch domain_path:=$domain problem_path:=$problem problem_name:=$problem_name results_folder:=$path &
# wait for the launch to start up
sleep 2
echo "Calling problem generation"
rosservice call /rosplan_problem_interface/problem_generation_server
sleep 3 
echo "Calling planner"
rosservice call /rosplan_planner_interface/planning_server
echo "Rename plan file and remove auto-generated problem"
echo "mv $path/plan.pddl $path/${problem_name}_plan.txt"
echo "rm ${problem}_problem.pddl"
mv $path/plan.pddl $path/${problem_name}_plan.txt
rm ${problem}_problem.pddl
sleep 3 
echo "Calling parser"
rosservice call /rosplan_parsing_interface/parse_plan
echo "Killing knowledge base for shutdown"
rosnode kill /rosplan_knowledge_base
wait $!
