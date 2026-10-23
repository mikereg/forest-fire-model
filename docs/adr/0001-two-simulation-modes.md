# Two simulation modes: stepwise and instantaneous

The literal Drossel–Schwabl rules (stepwise mode) let trees regrow beside a fire that is still burning, which violates the time-scale separation f ≪ p ≪ 1/T needed for the model to show critical, fractal behaviour; enforcing it within stepwise mode means p so small that collecting a fire-size distribution becomes impractically slow. We therefore implement both: stepwise mode for visualisation, and instantaneous mode — where a lightning strike burns its whole cluster in one time step — for all statistics. Both modes share the same lattice, neighbourhood, boundary and random-number handling, so the only difference between them is how burning is resolved.

## Considered Options

- **Stepwise only**: faithful to the 1992 rules as written, but fire-size statistics would be either contaminated by regrowth or prohibitively slow to gather.
- **Instantaneous only**: correct statistics, but there is no fire front to animate — the visualisation would show clusters vanishing in single frames.

## Consequences

A **fire** and its **fire size** are only measured in instantaneous mode. Stepwise mode reports no fire sizes, because concurrent fires can merge and a fire's size is then ill-defined.
