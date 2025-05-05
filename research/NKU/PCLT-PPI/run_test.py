import os


def run_func(description, ppi_path, pseq_path, vec_path, point_path, protein_max_length, index_path, gnn_model, test_all):
    os.system("python gnn_test.py \
            --description={} \
            --ppi_path={} \
            --pseq_path={} \
            --vec_path={} \
            --point_path={} \
            --protein_max_length={} \
            --index_path={} \
            --gnn_model={} \
            --test_all={} \
            ".format(description, ppi_path, pseq_path, vec_path, point_path, protein_max_length, index_path, gnn_model, test_all))


if __name__ == "__main__":
    description = "test"
    task = 'SHS27K'
    # task = 'STRING'

    # split_mode = "random_0.2"
    # split_mode = "bfs_0.2"
    split_mode = "dfs_0.2"

    # gnn_model = "save_model/SHS27k_random_11-19-01-45-56/gnn_model_train.ckpt"
    # gnn_model = "save_model/SHS27k_bfs_11-19-01-46-32/gnn_model_train.ckpt"
    gnn_model = "save_model/SHS27k_True1_" + split_mode + "_False/gnn_model_train.ckpt"

    ppi_path = "dataset/" + task + "/protein.actions." + task + ".STRING.txt"
    pseq_path = "dataset/" + task + "/protein." + task + ".sequences.dictionary.tsv"
    vec_path = "dataset/" + task + "/vec5_CTC.txt"
    point_path = "dataset/" + task + "/protein_point_dim/"
    index_path = "data_split/" + task + "_split/" + split_mode + ".txt"
    protein_max_length = 1024

    test_all = "True"
    # test_all = "False"

    # test test
    run_func(description, ppi_path, pseq_path, vec_path, point_path, protein_max_length, index_path, gnn_model, test_all)
