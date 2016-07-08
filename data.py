# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 20:55:38 2016

@author: Michael
"""

import pandas as pd
import numpy as np
from Data import data_init

class Workstation(object):
    def __init__(self, X=None, y=None, lamb=0, epsilon=0, hidden_layers=2):
        