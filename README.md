# gadf

done:

- GADFormer PyTorch Version (switch from PyTorch Lightning to reproducible, seed-stable (10seeds) and optimized PyTorch Version)
- scaler change robust to standard
- first improved experiment results:

  |setting |experiment  |AVG AUROC |AVG AUPRC |
  |--- | --- | ---| ---|
  |U|Synthetic Orig|0.94935+-0.033|0.81178+-0.104|
  |E|Synthetic Orig|0.98237+-0.020|0.93656+-0.053|
  |U|Synthetic Novelty .0|0.93292+-0.027|0.83512+-0.053|
  |U|Synthetic Novelty .01|0.96265+-0.024|0.82514+-0.067|

to be done:

- segment_len != 1
- add remaining experiment results
- update paper accordingly
- GRU re-run with 10 seeds
