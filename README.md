# Forest Fire Model

This Python script implements a simplified forest fire  model based on the [Drossel and Schwabl approach][1] (1992). The aim of this project is to graphically simulate the dynamics of a forest fire as governed  by the rules set out by Drossel and Schwabl and provide a mechanism for the further study of the critical behaviour associated with it under certain conditions.

## Table of Contents
* [Introduction](#introduction) 
* [Features](#features) 
* [Installation](#installation) 
* [Dependencies](#dependencies)
* [Contributing](#contributing) 
* [License](#license) 

## Introduction

A *Forest Fire Model* is a [cellular automaton](https://en.wikipedia.org/wiki/Cellular_automaton) on a $d$-dimensional hypercubic lattice with $L^d$ cells. Initially, each cell is either empty, occupied by a tree, or burning. As with all cellular automata, the state of the grid is updated at each time-step.

The Drossel and Schwabl model (1992) defines four rules, executed simultaneously, for updating the grid:
1.  A burning tree becomes an empty cell
2.  A tree will burn if at least one of its neighbours is burning
3.  A tree ignites with probability $f$ , even if no neighbours are burning
4. A tree grows in an empty cell with probability $p$.

[1]: https://physics.mcmaster.ca/~higgsp/756/DrosselSchwabl_ForestFire.pdf


## Features

In this simulation, the model is represented on a two-dimensional lattice. A summary of the design choices made  to implement the system in an intuitive fashion is described below:

### 	Parameters
- **Grid size (l)**: The length of the square grid used for the simulation
- **Probability of a tree catching fire (f)**: The user-defined probability that a tree-occupied cell turns into a burning cell
- **Probability of a tree growing in an empty cell \(p\)**: The user-defined probability that an empty cell becomes a tree-occupied cell.
- **Initial Configuration**: The script generates an initial random forest using the parameters specified above.

### Simulation Steps

1. **Initialisation**: Randomly generate the initial state of the forest.
2.  **Iterative simulation**: For each iteration, execute the rules set out above to update the grid. Strategic design considerations were made to allow for scalability - operations were vectorised where appropriate and neighbouring trees were defined using the [Moore Neighbourhood](https://en.wikipedia.org/wiki/Moore_neighborhood).
3. **Visualisation**: Display the evolving forest in real-time using matplotlib. (See below)

### Visualisation

The simulation is visualised using matplotlib, with each cell colour-coded to represent different cell states (black for empty, green for a tree-occupied cell, red for burning).  Here is an example of the simulation in action with $L = 200$, $p = 0.01$ and $f=0.0001$:

![forest fire gif](images/forest.gif)

## Installation

To run the simulation, follow these simple steps:
1. Clone the repository to your local machine
	```bash
	git clone https://github.com/mikereg/self-organised-criticality.git
	```
2.  Navigate to the appropriate directory

	```bash
	cd self-organised-criticality
	```
3. Run the script
	```bash
	python forest-fire.py
	```

## Dependencies
Make sure the following modules are installed prior to running the simulation:
* numpy
* matplotlib

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
