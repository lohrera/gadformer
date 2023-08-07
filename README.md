# gadf

done:

- GADFormer PyTorch Version (switch from PyTorch Lightning to reproducible, seed-stable (10seeds) and optimized PyTorch Version)
- scaler change robust to standard
- improved experiment results for unsupervised anomaly detection:

  |setting |experiment  |AVG AUROC |AVG AUPRC |
  |--- | --- | ---| ---|
  |U|Synthetic Orig|0.94935+-0.033|0.81178+-0.104|
  
- improved experiment results for semi-supervised anomaly detection:

  |setting |experiment  |AVG AUROC |AVG AUPRC |
  |--- | --- | ---| ---|
  |E|Synthetic Orig|0.98237+-0.020|0.93656+-0.053|


- improved experiment results for unsupervised novelty detection:

  |setting |experiment  |AVG AUROC |AVG AUPRC |
  |--- | --- | ---| ---|
  |U|Synthetic Novelty .0|0.93292+-0.027|0.83512+-0.053|
  |U|Synthetic Novelty .01|0.96265+-0.024|0.82514+-0.067|
  |U|Synthetic Novelty .05|0.93041+-0.030|0.68029+-0.102|

- improved experiment results for semi-supervised novelty detection:

  |setting |experiment  |AVG AUROC |AVG AUPRC |
  |--- | --- | ---| ---|
  |E|Synthetic Novelty .0|0.96895+-0.016|0.90605+-0.026|
  |E|Synthetic Novelty .01|0.97818+-0.006|0.89580+-0.031|
  |E|Synthetic Novelty .05|0.95834+-0.020|0.79626+-0.049|

- improved experiment results for unsupervised anomaly detection with noise:

  |setting |experiment  |AVG AUROC |AVG AUPRC |
  |--- | --- | ---| ---|
  |U|Synthetic Noise .0|0.93292+-0.027|0.83512+-0.053|
  |U|Synthetic Noise .2|0.90270+-0.033|0.75150+-0.092|
  |U|Synthetic Noise .5|0.78965+-0.061|0.44674+-0.195|

- improved experiment results for semi-supervised anomaly detection with noise:

  |setting |experiment  |AVG AUROC |AVG AUPRC |
  |--- | --- | ---| ---|
  |E|Synthetic Noise .0|0.96957+-0.016|0.90741+-0.026|
  |E|Synthetic Noise .2|0.95222+-0.018|0.87459+-0.023|
  |E|Synthetic Noise .5|0.89293+-0.021|0.73240+-0.054|

to be done:

- segment_len != 1
- add remaining experiment results
- update paper accordingly
- GRU re-run with 10 seeds
- permute
