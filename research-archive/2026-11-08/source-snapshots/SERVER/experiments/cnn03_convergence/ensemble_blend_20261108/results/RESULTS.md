# Ensemble and blend tune R

CPU float64. 384 tune queries. Confirm, full, and test were not scored.

## Reproduction

- curve_C_lr003_s0: 0.31019816453387611 (published 0.31019816453387611, abs delta 0.000e+00)
- curve_C_lr003_s1: 0.34520741683573092 (published 0.34520741683573092, abs delta 0.000e+00)
- curve_C_lr003_s2: 0.30980333573689856 (published 0.30980333573689856, abs delta 0.000e+00)
- hard_shared_control: 0.46211190598948326 (published 0.46211193532631023, abs delta 2.934e-08)

The control scalar missed 1e-12. Seeds 0, 1, and 2 matched with delta 0 on this same module, and the validation tensors match `house_edge.data`. An fp32 softmax path was 1.497e-08 away and was not used. Probabilities were not rescaled. The blend uses the canonical float64 control map. Weight 0 matches the seed-2 array exactly.

## Predeclared rows

- ensemble_equal_s0_s1_s2: 0.28975002607300254 (delta vs seed 2 -0.02005330966389601)
- ensemble_equal_s0_s2: 0.28468175920919014 (delta vs seed 2 -0.02512157652770841)
- blend_s2_control_w0.0: 0.30980333573689856 (delta vs seed 2 +0.00000000000000000)
- blend_s2_control_w0.025: 0.30693690199632961 (delta vs seed 2 -0.00286643374056894)
- blend_s2_control_w0.05: 0.30441273225560084 (delta vs seed 2 -0.00539060348129772)
- blend_s2_control_w0.1: 0.30039118477366389 (delta vs seed 2 -0.00941215096323467)
- blend_s2_control_w0.2: 0.29645525780787213 (delta vs seed 2 -0.01334807792902643)

## Ranking against the immutable current best

Current best tune R 0.30980333573689856. Threshold 1e-06.
Rows that clear 1e-6: ['ensemble_equal_s0_s2', 'ensemble_equal_s0_s1_s2', 'blend_s2_control_w0.2', 'blend_s2_control_w0.1', 'blend_s2_control_w0.05', 'blend_s2_control_w0.025'].
No package was written. ensemble_equal_s0_s2 is tune-informed. The blend grid was predeclared.
