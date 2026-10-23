# Forest-Fire Model

A simulation of the Drossel–Schwabl forest-fire model: a cellular automaton on a square lattice whose trees grow, are struck by lightning and burn, used to study self-organised criticality.

## Lattice

**Lattice**:
The square L × L grid of cells the model evolves on.
_Avoid_: board, map, forest (for the grid itself)

**Cell state**:
One of exactly three values a cell holds: **empty**, **tree** or **burning**.
_Avoid_: ash, fire (as a state), alive/dead

**Neighbourhood**:
The set of cells adjacent to a cell through which fire spreads; **von Neumann** (the 4 orthogonal cells, the default) or **Moore** (all 8 surrounding cells).

**Boundary**:
How cells on the edge of the lattice find their neighbours; **periodic** (the lattice wraps into a torus, the default) or **fixed** (cells beyond the edge count as empty).

**Cluster**:
A maximal set of tree cells connected through the neighbourhood.
_Avoid_: patch, group

**Tree density**:
The fraction of cells in the tree state.

## Dynamics

**Growth probability (p)**:
The probability per time step that an empty cell becomes a tree.

**Lightning probability (f)**:
The probability per time step that a tree ignites with no burning neighbour.
_Avoid_: ignition rate, fire probability

**Lightning strike**:
A single tree igniting spontaneously, with probability f.

**Time-scale separation**:
The regime f ≪ p ≪ 1/T, where T is the time the largest fire takes to burn out — fires are rare, and no trees grow while a fire burns. The regime in which the model is (approximately) critical.

## Simulation modes

**Stepwise mode**:
The four Drossel–Schwabl rules applied literally and simultaneously to every cell each time step, so fire advances one neighbourhood per step.
_Avoid_: real-time mode, animation mode

**Instantaneous mode**:
The time-scale-separated limit: a lightning strike burns its entire cluster within a single time step.
_Avoid_: fast mode, cluster mode

**Fire**:
The burning of one cluster struck by lightning in a time step; several strikes on the same cluster in one step make a single fire. Only well-defined in instantaneous mode; in stepwise mode fires can merge.

**Fire size**:
The number of trees burned in one fire.
_Avoid_: avalanche size, burn size, cluster size (a cluster exists before the fire; a fire size is a measurement)

## Analysis

**Fire-size distribution**:
The probability n(s) that a fire has size s, measured in the stationary state.

**Stationary state**:
The regime after transients have decayed, when tree density fluctuates around a constant mean.

**Exponent (τ)**:
The power-law exponent in n(s) ∝ s^(−τ), over the range of sizes where the distribution scales.
