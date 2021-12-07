# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:11:44 2021

@author: kpb20194
"""
import re

def getPlanDuration(filename):
    """ 
    This function parses plan text output and returns the duration of the plan as defined by the planner

    Input: filename - string representing filename of plan text file to be parsed.

    Ouput: integer value for duration of plan
    """
    try:
        with open(filename, "r", errors="ignore") as plan:
            for line in plan:
                if "; Cost" in line:
                    duration = float(re.search('\d+\.\d+', line).group())
            return duration
    except:
        return None
    