#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 14:44:49 2023

@author: balestrieri
"""

import dask
from time import sleep

data = [1, 2, 3, 4, 5, 6, 7, 8]


# Sequential code
def inc(x):
    sleep(1)
    return x + 1

# parallel code
@dask.delayed
def inc_par(x):
    sleep(1)
    return x+1


results = []
for x in data:
    y = inc(x)
    results.append(y)

total = sum(results)
print(total)

results_par = []
for x in data:
    y = inc_par(x)
    results_par.append(y)

total_par = sum(results_par)
print(total_par)



