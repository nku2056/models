import mindspore
import numpy as np

mindspore.set_context(device_target="Ascend")
device = mindspore.context.get_context('device_target')

# split_new = "True"
split_new = "False"

task = "SHS27k"
# task = "SHS148k"

# use_similarity = False
use_similarity = True
alpha = 0.01

split_mode = "random_0.2"
# split_mode = "bfs_0.2"
# split_mode = "dfs_0.2"

graph_only_train = "False"
# graph_only_train = "True"

description = task + "_" + split_mode
# description = task + "_" + "GCT" + "_" + split_mode

batch_size = 2048
epochs = 500
learning_rate = 1e-3
protein_max_length = 1024

ppi_path = "dataset/" + task + "/protein.actions." + task + ".STRING.txt"
pseq_path = "dataset/" + task + "/protein." + task + ".sequences.dictionary.tsv"
train_valid_index_path = "data_split/" + task + "_split/" + split_mode + ".txt"
use_lr_scheduler = "True"
save_path = "save_model/"
vec_path = "dataset/" + task + "/vec5_CTC.txt"
point_path = "dataset/" + task + "/protein_point_dim/"

similarity_matrix_path = "dataset/" + task + "/protein_similarity_" + task + ".txt"
similarity_matrix = np.loadtxt(similarity_matrix_path)
protein_protein_similarity_id = []
protein_protein_similarity_val = []
for i in range(similarity_matrix.shape[0]):
    for j in range(i):
        if similarity_matrix[i][j] > 0.5:
            protein_protein_similarity_id.append([j, i])
            protein_protein_similarity_val.append(similarity_matrix[i][j])


print(device)