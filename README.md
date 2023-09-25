# gadf

![GADFormer Architecture Overview](./GADFormer_Architecture_Overview.jpg?raw=true)

Abstract:
Group Anomaly Detection (GAD) identifies unusual pattern in groups where individual members might not be anomalous. This task is of major importance across multiple disciplines, in which also sequences like trajectories can be considered as a group. As groups become more diverse in heterogeneity and size, detecting group anomalies becomes challenging, especially without supervision. Though Recurrent Neural Networks are well established deep sequence models, their performance can decrease with increasing sequence lengths. 
Hence, this paper introduces GADformer, a BERT-based model for attention-driven GAD on trajectories in unsupervised and semi-supervised settings. We demonstrate how group anomalies can be detected by attention-based GAD. We also introduce the Block-Attention-anomaly-Score (BAS) to enhance model transparency by scoring attention patterns. In addition to that, synthetic trajectory generation allows various ablation studies. In extensive experiments we investigate our approach versus related works in their robustness for trajectory noise and novelties on synthetic data and three real world datasets.

requirements.txt:
```
torch==2.0.0
pandas==2.0.3
ruamel.yaml==0.17.32
python-stopwatch==1.0.5
matplotlib==3.6.2
scikit-learn==0.24.2
pytorch_optimizer==2.11.2
```

cli command:
```
python main_gadformer.py --root_dir="./datasets/"
```
