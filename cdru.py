# -*- coding: utf-8 -*-

from make_instances import makeSTN
import os
from PSTN_class import PSTN
import pickle as pkl
import numpy as np
import re
from parse_cctp import parse_cctp
inf = 10000

def main():
    dir_in = "pstns/domains/cdru/"
    dir_out = "pstns/problems/cdru/"
    for file in os.listdir(dir_in):
        parse_cctp(dir_in + file, dir_out + file[:-5])

if __name__ == '__main__':
    main()
