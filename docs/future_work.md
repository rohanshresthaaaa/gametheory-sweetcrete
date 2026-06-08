# Future Work — gametheory-sweetcrete

Maintained as a living list. Items flow directly from limitations found during the project.

## Data & experimental design
- **Decouple water/cement ratio from PCC level.** In the current data the recipe is fixed per
  PCC level, so w/c and PCC move together. Experiments that vary w/c *independently* would let
  the model learn their separate effects and sharpen the optimization.
- **More replicates and intermediate PCC levels** to reduce the replicate noise that currently
  caps cross-validated R² around 0.89.
- **Wastewater mixes.** The broader Sweetcrete project plans to replace fresh water with
  treated municipal/agricultural/industrial wastewater — add water-source as a feature and
  re-train.

## Modeling
- **Group-wise cross-validation** (group = mix × age) to estimate generalization to genuinely
  unseen mixes, complementing the current row-wise CV.
- **Uncertainty estimates** (e.g., quantile regression or conformal prediction) so the GUI can
  show a strength *range*, not just a point estimate — important for engineering safety margins.
- **Additional targets:** slump/workability and durability metrics, enabling true multi-target
  optimization.

## Optimization
- **Add a cost objective** (materials + cement price) to the game, turning the two-objective
  trade-off into a three-way one (cost, strength, sustainability).
- Explore other bargaining/compromise solutions (Kalai–Smorodinsky) alongside Nash.

## Validation & deployment
- **Field validation** against real pours (cf. the Burley, ID full-scale trial) to confirm
  lab-trained predictions hold in practice.
- Package the model + optimizer behind a small API so the GUI and field tools share one backend.
