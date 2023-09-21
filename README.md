# gadf

Abstract:
Group Anomaly Detection (GAD) identifies unusual pattern in groups where individual members might not be anomalous. This is crucial in fields where sequences, like trajectories, represent groups. As groups become more diverse in heterogeneity and size, detecting anomalies become challenging, especially without supervision. Though Recurrent Neural Networks handle sequences, their efficiency can drop with longer sequences. 
This paper presents GADformer, a BERT-based model tailored for attention-driven GAD on trajectories in unsupervised and semi-supervised settings. We demonstrate how trajectory anomalies can be detected using attention-based GAD. We also introduce the Block-Attention-Anomaly-Score (BAS) to enhance model transparency by scoring attention patterns. In addition to that, synthetic trajectory generation allows various ablation studies. In extensive experiments we investigate our approach versus related works in their robustness for trajectory noise and novelties on synthetic data and three real world datasets.

![GADFormer Architecture Overview](./GADFormer_Architecture_Overview.jpg?raw=true)
