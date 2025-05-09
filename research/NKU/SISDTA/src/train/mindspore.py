import mindspore
from mindspore import mint
import mindspore.dataset as ds
from mindspore import nn
from mindspore.experimental import optim

from tqdm import tqdm

from src.datasetm import MultiDatasetM
from src.model.gnnm import GNNM

def train_mindspore(args, fold):
    train = MultiDatasetM(args.dataset,
        device=args.device, sim_type=args.sim_type, setting=args.setting,
        d_threshold=args.d_threshold, p_threshold=args.p_threshold, fold=fold,
    )
    test = MultiDatasetM(args.dataset, train=False, 
        device=args.device, sim_type=args.sim_type, setting=args.setting,
        d_threshold=args.d_threshold, p_threshold=args.p_threshold, fold=fold,
    )
    column_names = ['dindex', 'pindex', 'd_vec', 'p_embedding', 'y']
    trainLoader = ds.GeneratorDataset(train, shuffle=True, column_names=column_names).batch(args.batch_size)
    testLoader = ds.GeneratorDataset(test, shuffle=False, column_names=column_names).batch(args.batch_size)

    mseLoss = nn.MSELoss()
    aeMseLoss = nn.MSELoss()
    model = GNNM(args.device, args.dropout)
    optimizer = optim.Adam(model.trainable_params(), lr=args.learning_rate, weight_decay=args.weight_decay)

    def forward_fn(d_index, p_index, d_vecs, p_embeddings, y):
        y_bar, decoded, feature = model(d_index, p_index, d_vecs, p_embeddings, train)
        mse = mseLoss(y, y_bar)
        loss = mse + args.lambda_1 * aeMseLoss(decoded, feature)
        return loss, y_bar, decoded, feature

    print('training fold {}...'.format(fold))
    for epoch in range(1, args.epochs + 1):
        model.set_train()
        for d_index, p_index, d_vecs, p_embeddings, y in tqdm(trainLoader, leave=False):
            grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
            (trainLoss, y_bar, decoded, feature), grads = grad_fn(d_index, p_index, d_vecs, p_embeddings, y)
            trainMse = mseLoss(y, y_bar)
            optimizer(grads)
        
        if epoch % 10 != 0 and epoch != args.epochs: continue

        count, testMse = 0, 0
        model.set_train(False)
        for d_index, p_index, d_vecs, p_embeddings, y in testLoader:
            y_bar, _, _, = model(d_index, p_index, d_vecs, p_embeddings, test)
            testMse += mseLoss(y, y_bar)
            count += 1

        result = 'Fold: {} Epoch: {} train loss: {:.6f} train mse: {:.6f} test_mse: {:.6f}'.format(fold, epoch, trainLoss.asnumpy(), trainMse.item(), testMse.item() / count)
        print(result)

    return result
