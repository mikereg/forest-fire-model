#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 22:51:15 2023

@author: michaelreginiano
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.pyplot import draw, pause


l = 200 #length of grid

#generating initial random forest
grid = np.random.randint(0,2,(l,l))

f = 0.0001
p = 0.01

def moore_neighbourhood(x,y,grid):
    x_lower,x_upper = 1,1
    y_lower,y_upper = 1,1
    GRID_LENGTH = grid.shape[0]
    
    if (x==0) & (y==0):
        x_lower,y_lower=0,0
    elif x==0:
        x_lower=0
    elif y==0:
        y_lower=0
    
    if (x==GRID_LENGTH) & (y==GRID_LENGTH):
        x_upper,y_upper=0,0
    elif x==GRID_LENGTH:
        x_upper=0
    elif y==GRID_LENGTH:
        y_upper=0
    
    row_start = x-x_lower
    row_end = x+x_upper+1
    
    col_start = y-y_lower
    col_end = y+y_upper+1

    return grid[row_start:row_end,col_start:col_end]


fig = plt.figure()
ax = fig.gca()

colors_list = ['black','green','red']
cmap = colors.ListedColormap(colors_list)
board_image = ax.imshow(grid,cmap=cmap,vmin=0,vmax=2)


for _ in range(0,200):

    grid_old = grid.copy()
    board_image.set_data(grid)
    board_image.norm.autoscale([0,1,2])
    draw()
    pause(0.05)
    
    grid[grid_old==2] = 0 

    x,y = np.where(grid_old == 1)[0], np.where(grid_old == 1)[1]
    trees = np.array([(x[i],y[i]) for i in range(0,len(x))], dtype=tuple)

    grid[grid_old == 1] = np.random.choice([1,2],len(trees),p=[1-f,f])
   
    grid[grid_old == 0] = np.random.choice([0,1],np.sum(grid_old == 0),p=[1-p,p])

    for t in trees:
        if any((moore_neighbourhood(t[0], t[1], grid_old) == 2).flatten()):
            grid[t[0],t[1]] = 2
