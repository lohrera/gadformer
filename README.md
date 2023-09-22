# gadf

Abstract:
Group Anomaly Detection (GAD) identifies unusual pattern in groups where individual members might not be anomalous. This task is of major importance across multiple disciplines, in which also sequences like trajectories can be considered as a group. As groups become more diverse in heterogeneity and size, detecting group anomalies becomes challenging, especially without supervision. Though Recurrent Neural Networks are well established deep sequence models, their performance can decrease with increasing sequence lengths. 
Hence, this paper introduces GADformer, a BERT-based model for attention-driven GAD on trajectories in unsupervised and semi-supervised settings. We demonstrate how group anomalies can be detected by attention-based GAD. We also introduce the Block-Attention-anomaly-Score (BAS) to enhance model transparency by scoring attention patterns. In addition to that, synthetic trajectory generation allows various ablation studies. In extensive experiments we investigate our approach versus related works in their robustness for trajectory noise and novelties on synthetic data and three real world datasets.

![GADFormer Architecture Overview](./GADFormer_Architecture_Overview.jpg?raw=true)
