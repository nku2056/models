
## PointPPI

Here is the code implemented using Mindspore for paper PCLT-PPI: Predicting Multi-type Interactions between Proteins based on Point Cloud Structure and Local Topology Preservation.

### Innovation

* Compared to other methods that use protein structure information, PCLT-PPI directly utilizes point cloud structure, which can extract more comprehensive protein structure information.

* Compared to other existing methods, PCLT-PPI effectively maintains the local topology of proteins in embedding spaces, which can help the model learn better.

### Dataset

* All proteins used in the experiment were sourced from the String dataset, with a total of 15335 proteins. Only 20 common amino acids were retained for each protein

* The three-dimensional structure of proteins is determined by [Alphafold](https://alphafold.ebi.ac.uk/download) Predicted (including direct download and manual prediction)

### Train & Test

* Training codes in gnn_train.py, and the run script in run.py.

* Testing codes in gnn_test.py , and the run script in run_test.py.

Terminal Command: python gnn_train.py --task  --ppi_path   --pseq_path  --vec_path  --point_path  --protein_max_length  --batch_size  --epochs  --split_new --split_mode --train_valid_index_path --save_path  --graph_only_train  --train_valid_index_path