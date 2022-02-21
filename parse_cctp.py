# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 16:11:06 2021

@author: kpb20194
"""
from time import time
import xml.etree.ElementTree
import PSTN_class as PSTN
import numpy as np
import pickle as pkl

inf = 10000

def parse_cctp(filename, out):
    """ 
    This function parses .cctp files and outputs an PSTN.

    Input: filename - string representing filename of dot file to be parsed.

    Ouput: List of dictionaries representing an STN. Each item in the list is a dictionary representing a constraint between
    two time points.
    """
    e = xml.etree.ElementTree.parse(filename).getroot()

    problem_name = e.find('NAME').text

    # In TPN format every node (or event in TPN terminology) has a non-unique name
    # and an unique id. Both of those are strings. For efficiency DC checking algorithms
    # denote each node by a number, such that we can cheaply check their equality.

    # parse the event
    timepoints = []
    for event_obj in e.findall('EVENT'):
        e_name, e_description = event_obj.find('ID').text, event_obj.find('NAME').text
        timepoints.append(PSTN.timePoint(e_name, e_description))
    

    # parse the temporal constraints
    constraints = []
    # if line below confuses you, that's expected... We need better automated code generation for parsing...
    for constraint_obj in e.findall('CONSTRAINT'):
        # duration can be one of three types - controllable, uncertain and probabilistic
        lower_bound = float(constraint_obj.find('LOWERBOUND').text)
        upper_bound = float(constraint_obj.find('UPPERBOUND').text)

        from_event = constraint_obj.find('START').text
        to_event = constraint_obj.find('END').text
        #constraint_id = constraint_obj.find('ID').text
        constraint_name = constraint_obj.find('NAME').text

        #(self, description, source, sink, type, intervals, distribution = None, hard = True)
        # check if the constraint is controllable and relaxable
        if constraint_obj.find('MEAN') is not None and constraint_obj.find('VARIANCE') is not None:
            dist = {"type": "gaussian", "mean": float(constraint_obj.find('MEAN').text), "variance": np.sqrt(float(constraint_obj.find('VARIANCE').text))}
            for timepoint in timepoints:
                if timepoint.id == from_event:
                    source = timepoint
                elif timepoint.id == to_event:
                    sink = timepoint
            constraints.append(PSTN.constraint(constraint_name, source, sink, "pstc", intervals = {"lb": 0, "ub": inf, "value": 1}, distribution=dist))
        
        else:
            for timepoint in timepoints:
                if timepoint.id == from_event:
                    source = timepoint
                elif timepoint.id == to_event:
                    sink = timepoint
            
            if constraint_obj.find('LBRELAXABLE') is not None and constraint_obj.find('UBRELAXABLE') is not None:
                if "T" in constraint_obj.find('LBRELAXABLE').text:
                    hard = False
                else:
                    hard = True
            constraints.append(PSTN.constraint(constraint_name, source, sink, "stc", intervals = {"lb": lower_bound, "ub": upper_bound, "value": 1}))
    instance = PSTN.PSTN(problem_name, timepoints, constraints)
    
    print(out)
    with open(out, "wb") as f:
            pkl.dump(instance, f)
        