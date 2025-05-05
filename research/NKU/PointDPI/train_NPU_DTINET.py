import json
import random
import argparse
import pickle
from mindspore import ops,save_checkpoint
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import context
from mindspore.ops import operations as P
from mindspore.nn import WithLossCell, TrainOneStepCell
from utils_for_matching_dtinet import *
from datetime import datetime
import sys
from c2net.context import prepare
from c2net.context import upload_output
c2net_context = prepare()
datasetPath = c2net_context.dataset_path
codePath = c2net_context.code_path
outputPath = c2net_context.output_path


#输出内容到文件中
# file = open(outputPath+"/train.txt", "a+")
# original_stdout = sys.stdout
# sys.stdout = file

now = datetime.now()
formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
print(formatted_now)
#mindsopre设置GPU
context.set_context(mode=context.PYNATIVE_MODE,device_target="Ascend")
#squeeze
squeeze = P.Squeeze()

#################基础设置################
LR = 0.0001
SEED = 3142
# VAL_SIZE = 7486
BATCH_SIZE = 128
EPOCH = 500
Feature_Size = 256   # [256, 512]
Alpha_d = 1
Alpha_p = 1

#使用dump好的pkl
use_pkl = True

###############药物编码模块###############
# SMILES_Coding
DSC_Kernel_Num = 32
DSC_Kernel_Size = 8
Drug_SMILES_Input_Size = 128      # [128, 256]

# Image_Coding
Drug_Point_Hidden_Size = 512   # [128, 256, 512]
DPC_Kernel_Num = 32
DPC_Kernel_Size = 8   # [8, 16]

###############蛋白编码模块###############
# Bert_Coding
Protein_Bert_Hidden_Size = 512

#Point_Coding
Protein_Point_Hidden_Size = 512   # [128, 256, 512]
PPC_Kernel_Num = 32
PPC_Kernel_Size = 8   # [8, 16]

###############数据处理设置################
Drug_Max_Lengtgh = 100
Protein_Max_Lengtgh = 1024
AA_Dict = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'B', 'Z']
Protein_Dic_Length = 23
Atom_Point_Dict_Length = 79
atom_dict = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, "1": 35, "0": 3,
            "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, "9": 39, "8": 7, "=": 40, "A": 41,
            "@": 8, "C": 42, "B": 9, "E": 43, "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46,
            "M": 47, "L": 13, "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
            "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, "b": 21, "e": 57,
            "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, "l": 25, "o": 61, "n": 26, "s": 62,
            "r": 27, "u": 63, "t": 28, "y": 64}

Atom_Dic_Length = 64

mm = ops.MatMul()

# def einsum_ops(x1,x2):
#     x = ms.Tensor(np.empty((1,x1.shape[1],x2.shape[2]),dtype='float32'))
#     concat_op = ops.Concat(axis=0)
#     matmul = ops.MatMul()
#     for i in range(x1.shape[0]):
#         # x3 = matmul(x1[i], x2[i])
#         # x3 = ms.Tensor(np.expand_dims(x3.asnumpy(),axis=0))
#
#         x3 = matmul(x1[i], x2[i])
#         expand_dims = ops.ExpandDims()
#         x3 = expand_dims(x3, 0)
#         x = concat_op(x,x3)
#     return x[1:,:,:]

# def einsum_ops(x1, x2):
#     zeros = ops.Zeros()
#     x = zeros((1, x1.shape[1], x2.shape[2]), ms.float32)
#     concat_op = ops.Concat(axis=0)
#     matmul = ops.MatMul()
#     for i in range(x1.shape[0]):
#         x3 = matmul(x1[i], x2[i])
#         expand_dims = ops.ExpandDims()
#         x3 = expand_dims(x3, 0)
#         x = concat_op(x, x3)
#     return x[1:,:,:]

def einsum_ops(x1, x2):
    batmatmul = ops.BatchMatMul()
    return batmatmul(x1,x2)

equation = "ijk, ikp->ijp"


class GraphConvolution(nn.Cell):
    def __init__(self, in_size, out_size,):
        super(GraphConvolution, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = ms.Parameter(ms.Tensor(in_size, out_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def construct(self, x, a):
        support = mm(x, self.weight)  # X*W
        r = mm(a, support)    # A*X*W
        return r


class DrugSMILESCoding(nn.Cell):
    def __init__(self, hid_dim=Drug_SMILES_Input_Size, out_dim=Feature_Size, vocab_size=Atom_Dic_Length,
                 channel=DSC_Kernel_Num, kernel_size=DSC_Kernel_Size):
        super(DrugSMILESCoding, self).__init__()
        self.embedding = \
            nn.Embedding(vocab_size, embedding_size=hid_dim)
        self.conv1 = nn.Conv1d(hid_dim, channel, kernel_size,pad_mode='pad', padding=kernel_size - 1)
        self.conv2 = nn.Conv1d(channel, channel * 2, kernel_size,pad_mode='pad', padding=kernel_size - 1)
        self.conv3 = nn.Conv1d(channel * 2, channel * 4, kernel_size,pad_mode='pad', padding=kernel_size - 1)
        # self.conv1 = nn.Conv1d(hid_dim, channel, kernel_size, padding=kernel_size-1)
        # self.conv2 = nn.Conv1d(channel, channel*2, kernel_size, padding=kernel_size-1)
        # self.conv3 = nn.Conv1d(channel*2, channel*4, kernel_size, padding=kernel_size-1)
        self.act = nn.LeakyReLU(0.2)
        self.globalmaxpool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Dense(channel*4, out_dim)

    def construct(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.act(x)
        x = self.globalmaxpool(x)
        x = x.squeeze(-1)
        x = self.fc1(x)
        return x



class DrugPointCoding(nn.Cell):
    def __init__(self, point_hid_dim=Drug_Point_Hidden_Size, point_output_dim=Feature_Size,
                 channel=DPC_Kernel_Num, kernel_size=DPC_Kernel_Size):
        super(DrugPointCoding, self).__init__()
        self.gcn1 = nn.SequentialCell(
            nn.Embedding(Atom_Point_Dict_Length, point_hid_dim),
            nn.Dense(point_hid_dim, point_hid_dim))
        self.gcn2 = nn.Dense(point_hid_dim, point_output_dim)

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)

        self.conv = nn.SequentialCell(
            nn.Conv1d(point_output_dim, channel, kernel_size,pad_mode='pad', padding=kernel_size - 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channel, channel * 2, kernel_size,pad_mode='pad', padding=kernel_size - 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channel * 2, channel * 4, kernel_size,pad_mode='pad', padding=kernel_size - 1),
            nn.LeakyReLU(0.2),
        )#增加pad_mode='pad'
        self.act = nn.LeakyReLU(0.2)
        self.globalmaxpool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Dense(channel * 4, point_output_dim)
        self.act = nn.LeakyReLU(0.2)

    def construct(self, xd):
        d_d = xd[:, :, :xd.shape[1]]  # 距离
        d_c = xd[:, :, xd.shape[1]: xd.shape[1]*2]  # 距离
        d_a = xd[:, :, -1]  # 特征
        d_a = squeeze(d_a).long()

        h1 = self.gcn1(d_a)
        # einsum_op = ops.Einsum('ijk, ikp->ijp')
        # h1_1 = einsum_op([d_d, h1]) # ops.Einsum('ijk, ikp->ijp', [d_c, h1])
        # h1_2 = einsum_op([d_c, h1])
        d_d = ms.Tensor(d_d)
        h1 = ms.Tensor(h1)
        h1_1 = einsum_ops(d_d,h1)
        # h1_1 = ops.einsum(equation,d_d,h1)

        # h1_1 = np.einsum('ijk, ikp->ijp',d_d,h1)
        # h1_2 = np.einsum('ijk, ikp->ijp',d_c,h1)
        h1_2 = einsum_ops(d_c,h1)
        # h1_2 = ops.einsum(equation,d_c,h1)


        h1_1 = ms.Tensor(h1_1)
        h1_2 = ms.Tensor(h1_2)
        # print("===="+h1_1.shape)
        b, r, c = h1_1.shape
        # b = h1_1.shape[0]
        # r = h1_1.shape[1]
        # c = h1_1.shape[2]
        h1_1 = self.act(h1_1).view(b, 1, r, c)
        h1_2 = self.act(h1_2).view(b, 1, r, c)
        h1_12 = ops.cat((h1_1, h1_2), axis=1)
        h1_c = self.conv1(h1_12).view(b, r, c)
        h1 = self.act(h1_c)

        h2 = self.gcn2(h1)
        # h2_1 = einsum_op([d_d, h2])
        # h2_2 = einsum_op([d_c, h2])
        h2_1 = einsum_ops(d_d, h2)
        # h2_1 = ops.einsum(equation,d_d,h2)

        h2_2 = einsum_ops(d_c, h2)
        # h2_2 = ops.einsum(equation,d_c,h2)

        # h2_1 = np.einsum('ijk, ikp->ijp',d_d, h2)
        # h2_2 = np.einsum('ijk, ikp->ijp',d_c, h2)

        b, r, c = h2_1.shape
        h2_1 = self.act(h2_1).view(b, 1, r, c)
        h2_2 = self.act(h2_2).view(b, 1, r, c)
        h2_12 = ops.cat((h2_1, h2_2), axis=1)
        h2_c = self.conv2(h2_12).view(b, r, c)
        h2 = self.act(h2_c)
        h2 = self.act(h2)
        x = h2.permute(0, 2, 1)
        x = self.conv(x)
        x = self.globalmaxpool(x)
        x = x.squeeze(-1)
        x = self.fc1(x)
        return x


class DrugCoding(nn.Cell):
    def __init__(self):
        super(DrugCoding, self).__init__()
        self.coding1 = DrugSMILESCoding()
        self.coding2 = DrugPointCoding()

    def construct(self, x_smiles, x_image):
        e_graph = self.coding1(x_smiles)
        e_image = self.coding2(x_image)

        return e_graph, e_image


class ProteinBertCoding(nn.Cell):
    def __init__(self, bert_hid_dim=Protein_Bert_Hidden_Size, bert_output_dim=Feature_Size):
        super(ProteinBertCoding, self).__init__()
        self.seq_coding = nn.SequentialCell(
            nn.Dense(1024, bert_hid_dim),
            nn.LeakyReLU(0.2),
            nn.Dense(bert_hid_dim, bert_output_dim),
            nn.Sigmoid(),
        )

    def construct(self, xp):
        ep = self.seq_coding(xp)
        return ep


class ProteinPointCoding(nn.Cell):
    def __init__(self, point_hid_dim=Protein_Point_Hidden_Size, point_output_dim=Feature_Size,
                 channel=PPC_Kernel_Num, kernel_size=PPC_Kernel_Size):
        super(ProteinPointCoding, self).__init__()
        self.gcn1 = nn.SequentialCell(
            nn.Embedding(len(AA_Dict) + 1, point_hid_dim),
            nn.Dense(point_hid_dim, point_hid_dim))
        self.gcn2 = nn.Dense(point_hid_dim, point_output_dim)

        self.conv1 = nn.SequentialCell(
            nn.Conv1d(point_output_dim, channel, kernel_size,pad_mode='pad', padding=kernel_size-1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channel, channel * 2, kernel_size,pad_mode='pad', padding=kernel_size-1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channel * 2, channel * 4, kernel_size,pad_mode='pad', padding=kernel_size-1),
            nn.LeakyReLU(0.2),
        )#增加pad_mode='pad'
        self.act = nn.LeakyReLU(0.2)
        self.globalmaxpool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Dense(channel * 4, point_output_dim)
        self.act = nn.LeakyReLU(0.2)

    def construct(self, xp):
        p_t = xp[:, :, :xp.shape[1]]  # 拓扑
        p_a = xp[:, :, xp.shape[1]:]  # 特征

        # p_a = ops.Squeeze(p_a).long()
        p_a = squeeze(p_a).long()
        h1 = self.gcn1(p_a)
        # einsum_op = ops.Einsum('ijk, ikp->ijp')
        # h1 = einsum_op([p_t, h1])
        h1 = einsum_ops(p_t,h1)
        # h1 = ops.einsum(equation,p_t,h1)


        # h1 = ops.Einsum('ijk, ikp->ijp', [p_t, h1])
        h1 = self.act(h1)
        h2 = self.gcn2(h1)
        h2 = einsum_ops(p_t,h2)
        # h2 = ops.einsum(equation,p_t,h2)

        # h2 = einsum_op([p_t, h2])
        # h2 = ops.Einsum('ijk, ikp->ijp', [p_t, h2])
        h2 = self.act(h2)
        x = h2.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.globalmaxpool(x)
        x = x.squeeze(-1)
        x = self.fc1(x)
        return x


class ProteinCoding(nn.Cell):
    def __init__(self):
        super(ProteinCoding, self).__init__()
        self.coding1 = ProteinBertCoding()
        self.coding2 = ProteinPointCoding()

    def construct(self, x_bert, x_point):
        e_bert = self.coding1(x_bert)
        e_point = self.coding2(x_point)
        return e_bert, e_point


class PreNetMLP(nn.Cell):
    def __init__(self, smiles_output_dim=Feature_Size, bert_output_dim=Feature_Size):
        super(PreNetMLP, self).__init__()
        self.d_c = DrugCoding()
        self.p_c = ProteinCoding()
        self.fc1 = nn.Dense(smiles_output_dim + bert_output_dim, 1024)
        self.fc2 = nn.Dense(1024, 256)
        self.fc3 = nn.Dense(256, 2)
        self.act1 = nn.LeakyReLU(0.2)
        self.act2 = nn.Tanh()

    def construct(self, d_s, d_i, p_b, p_p):
        eds, edi = self.d_c(d_s, d_i)
        epb, epp = self.p_c(p_b, p_p)
        e = ops.cat((eds, epp), axis=1)
        s0 = self.fc1(e)
        a0 = self.act1(s0)
        s1 = self.fc2(a0)
        a1 = self.act2(s1)
        s2 = self.fc3(a1)
        return eds, edi, epb, epp, s2


def seed_torch(seed):
    random.seed()
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    ms.set_seed(seed) #torch.manual_seed(seed)


class MyGenerator:
    def __init__(self, drug_idx, protein_idx, label):
        self.drug_idx = drug_idx
        self.protein_idx = protein_idx
        self.label = label

    def __iter__(self):
        for d, p, l in zip(self.drug_idx, self.protein_idx, self.label):
            yield (d, p, l)


# mindspore带有损失函数的class
class MultipleLossCell(WithLossCell):
    def __init__(self, backbone, loss_fn1, loss_fn2):
        super(MultipleLossCell, self).__init__(backbone,loss_fn1)
        self._backbone = backbone
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2

    def construct(self, batch_xd1, batch_xd2, batch_xp1, batch_xp2, batch_y):
        batch_ed1, batch_ed2, batch_ep1, batch_ep2, batch_pre = self._backbone(batch_xd1, batch_xd2, batch_xp1, batch_xp2)
        loss1 = self.loss_fn1(batch_ed1, batch_ed2)
        loss2 = self.loss_fn1(batch_ep1, batch_ep2)
        loss3 = self.loss_fn2(batch_pre, batch_y)
        loss = Alpha_d * loss1 + Alpha_p * loss2 + loss3
        return loss
    def backbone_network(self):
        return self._backbone

        
if __name__ == '__main__':
    seed_torch(SEED)
    # task = 'drugbank'   # select from 'drugbank', 'bindingdb' and 'dtinet'
    # task = 'dtinet'  # select from 'drugbank', 'bindingdb' and 'dtinet'
    # task = 'bindingdb'  # select from 'drugbank', 'bindingdb' and 'dtinet'
    # confi = 'confi60'
    parser = argparse.ArgumentParser(description='train and test set')
    parser.add_argument('--task', type=str, default='dtinet', help='task name')
    parser.add_argument('--path', type=str, default='', help='file name')

    #适配NPU镜像
    parser.add_argument('--grampus_code_file_name', type=str, help='Grampus code file name')
    parser.add_argument('--model_url', type=str, help='Model URL')
    parser.add_argument('--multi_data_url', type=str, help='Multi data URL')
    parser.add_argument('--pretrain_url', type=str, help='Pretrain URL')
    parser.add_argument('--data_url', type=str, help='Data URL')
    parser.add_argument('--train_url', type=str, help='Train URL')
    parser.add_argument('--result_url', type=str, help='Result URL')
    parser.add_argument('--device_target', type=str, help='Device Target')

    args = parser.parse_args()
    # 解析dataset
    # data_urls = json.loads(args.multi_data_url)
    # datasetPath = args.data_url
    # filename = data_urls['dataset_name']

    # outputPath = args.result_url
    task = args.task
    path = args.path
    # 将所有的药物和蛋白所需数据读入内存中方便后续快速取用，确保针对每种药物/蛋白只处理一次


    if use_pkl:
        with open(datasetPath+'/'+filename+'/' + task +'/drug_smiles_data.pkl', 'rb') as f:
            drug_smiles_data = pickle.load(f)
        with open(datasetPath+'/' +filename+'/' + task +'/drug_points_data.pkl', 'rb') as f:
            drug_points_data = pickle.load(f)
        with open(datasetPath+'/' +filename+'/' +  task +'/protein_bert_data.pkl', 'rb') as f:
            protein_bert_data = pickle.load(f)
        with open(datasetPath+'/' +filename+'/' +  task +'/protein_point_data.pkl', 'rb') as f:
            protein_point_data = pickle.load(f)
    else:
        print("dump")
        drug_smiles_data, drug_points_data = data_preparation_drug_all(task)
        # with open(task + '/drug_smiles_data.pkl', 'wb') as f:
        #     pickle.dump(drug_smiles_data, f)
        # with open(task + '/drug_points_data.pkl', 'wb') as f:
        #     pickle.dump(drug_points_data, f)
        protein_bert_data, protein_point_data = data_preparation_protein_all(task)
        # with open(task + '/protein_bert_data.pkl', 'wb') as f:
        #     pickle.dump(protein_bert_data, f, protocol=4)
        # with open(task + '/protein_point_data.pkl', 'wb') as f:
        #     pickle.dump(protein_point_data, f, protocol=4)


    # with open(task+'/protein_bert_data.pkl', 'rb') as f:
    #     protein_bert_data = pickle.load(f)
    # with open(task+'/protein_point_data.pkl', 'rb') as f:
    #     protein_point_data = pickle.load(f)

    with open(codePath + '/pointdpi/dataset/' + task + '/result/train' + path + '.csv') as f1:
        train_data = f1.readlines()
    num_sample = len(train_data)
    drug_idx_train, protein_idx_train, label_train = data_preparation(train_data, task)

    generator = MyGenerator(drug_idx_train, protein_idx_train, label_train)
    dataset = ds.GeneratorDataset(source=generator, column_names=['drug_idx', 'protein_idx', 'label'])

    dataset = dataset.shuffle(buffer_size=dataset.get_dataset_size())
    dataset = dataset.batch(batch_size=BATCH_SIZE)

    # dataset = TensorDataset(drug_idx_train, protein_idx_train, label_train)
    # dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    model = PreNetMLP()
    # print(model)
    loss_func1 = nn.MSELoss()
    loss_func2 = nn.CrossEntropyLoss()
    params = model.trainable_params()
    optim = nn.Adam(params, learning_rate=LR)#optim = torch.optim.Adam(model.parameters(), lr=LR)
    model_with_loss = MultipleLossCell(model, loss_func1, loss_func2)
    train_network = TrainOneStepCell(model_with_loss,optim)
    train_network.set_train()
    print('Start training...')
    max_auc = 0
    max_aupr = 0
    for epoch in range(EPOCH):
        for step, data in enumerate(dataset):
            batch_d_idx, batch_p_idx, batch_y = data
            batch_xd1, batch_xd2 = data_preparation_drug(drug_smiles_data, drug_points_data, batch_d_idx)  # xd1存序列，xd2存分子图片
            batch_xp1, batch_xp2 = data_preparation_protein(protein_bert_data, protein_point_data, batch_p_idx)  # xp1存序列特征，xp2存3d点云
            batch_y = batch_y.astype(ms.int32)
            loss = train_network(batch_xd1, batch_xd2, batch_xp1, batch_xp2,batch_y)
            # print(step)

            # if torch.cuda.is_available():
            #         batch_xd1 = batch_xd1.cuda()
            #         batch_xd2 = batch_xd2.cuda()
            #         batch_xp1 = batch_xp1.cuda()
            #         batch_xp2 = batch_xp2.cuda()
            #         batch_y = batch_y.cuda()
            # print(epoch, step)
            # batch_ed1, batch_ed2, batch_ep1, batch_ep2, batch_pre = model(batch_xd1, batch_xd2, batch_xp1, batch_xp2)
            # loss1 = loss_func1(batch_ed1, batch_ed2)
            # loss2 = loss_func1(batch_ep1, batch_ep2)
            # batch_y = batch_y.astype(ms.int32)
            # loss3 = loss_func2(batch_pre, batch_y)
            # loss = Alpha_d * loss1 + Alpha_p * loss2 + loss3
            # optim.zero_grad()
            # loss.backward()
            # optim.step()
            if step % 10 == 0:
            #     # if torch.cuda.is_available():
            #     #     print('Epoch: ', epoch, ' | Step: ', step, '/', int(num_sample / BATCH_SIZE) + 1,
            #     #           '| loss: %.20f' % loss3.cpu().item(), '| loss_d: %.20f' % loss1.cpu().item(),
            #     #           '| loss_p: %.20f' % loss2.cpu().item())
            #     # else:
                print('Epoch: ', epoch,' | Step: ', step, '| loss: %.20f' % loss.item())
                # file.flush()

        with open(codePath + '/pointdpi/dataset/' + task + '/result/test' + path + '.csv') as f1:
            val_data = f1.readlines()
        # num_sample = len(train_data)
        drug_idx_val, protein_idx_val, label_val = data_preparation(val_data, task)

        # dataset_val = TensorDataset(drug_idx_val, protein_idx_val, label_val)
        val_generator = MyGenerator(drug_idx_val, protein_idx_val, label_val)
        dataset_val = ds.GeneratorDataset(source=val_generator, column_names=['drug_idx', 'protein_idx', 'label'])
        dataset_val = dataset_val.shuffle(buffer_size=dataset_val.get_dataset_size())
        dataset_val = dataset_val.batch(batch_size=BATCH_SIZE)
        # dataloader_val = DataLoader(dataset=dataset_val, batch_size=BATCH_SIZE, shuffle=False)
        pre_val, y_val = [], []
        for step_val, data_val in enumerate(dataset_val):
            batch_d_idx_val, batch_p_idx_val, batch_y_val = data_val
            batch_xd1_val, batch_xd2_val = data_preparation_drug(drug_smiles_data, drug_points_data,
                                                                 batch_d_idx_val)  # xd1存序列，xd2存分子图片
            batch_xp1_val, batch_xp2_val = data_preparation_protein(protein_bert_data, protein_point_data,
                                                                    batch_p_idx_val)  # xp1存序列特征，xp2存3d点云

            # if torch.cuda.is_available():
            #     batch_xd1_val = batch_xd1_val.cuda()
            #     batch_xd2_val = batch_xd2_val.cuda()
            #     batch_xp1_val = batch_xp1_val.cuda()
            #     batch_xp2_val = batch_xp2_val.cuda()
            #     batch_y_val = batch_y_val.cuda()
            # print(epoch, step)
            batch_ed1_val, batch_ed2_val, batch_ep1_val, batch_ep2_val, batch_pre_val = model(batch_xd1_val,
                                                                                              batch_xd2_val,
                                                                                              batch_xp1_val,
                                                                                              batch_xp2_val)
            # pre_val += batch_pre_val.detach().cpu().numpy()[:, 1].tolist()
            # y_val += batch_y_val.detach().cpu().numpy().tolist()
            pre_val += batch_pre_val[:, 1].tolist()
            y_val += batch_y_val.tolist()
        val_metrics = val_evalute(pre_val, y_val)  # auc, aupr, acc, sen, spe
        print('AUC = ', str(val_metrics[0]), ' | AUPR = ', str(val_metrics[1]))
        if val_metrics[0] > max_auc and val_metrics[1] > max_aupr:
            max_auc = val_metrics[0]
            max_aupr = val_metrics[1]
            max_acc = val_metrics[2]
            max_sen = val_metrics[3]
            max_spe = val_metrics[4]
            # torch.save(model.state_dict(),
            #            'models/best_Matching_' + task + '_AUC' + str(max_auc) + '_' + '_seed' + str(SEED) + '.pth')
            print('Get a better performance! Max_AUC = ' + str(max_auc) + ' and Max_AUPR = ' + str(
                val_metrics[1]) + 'ACC SEN SPE = ' + str(max_acc) + str(max_sen) + str(max_spe))
            # file.flush()
            # torch.save(model.state_dict(),
            #            'models/best_Matching_' + task + '_' + path + '_' + str(max_auc) + '_' + '_seed' + str(
            #                SEED) + '.pth')
            # save_checkpoint(model,
            #                 "mindspore_result"+'/best_Matching_' + task + '_' + path + '_' + str(max_auc) + '_' + '_seed' + str(
            #                     SEED) + '.ckpt')
            # upload_output()

