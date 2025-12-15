from __future__ import print_function, division
import argparse
from email.policy import default
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import numpy as np
import torch
from itertools import chain
import torch_clustering
import os
import logging
import random
from Network import Network
import torch.nn.functional as F
from clusteringPerformance import clusteringMetrics
from Dataloader import MultiViewDatasetLoader
from torch.utils.data import DataLoader
from utils import CrossEn,KL

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configure logging
def setup_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s'))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)



def main():
    # Ensure environment variables and other setup
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    use_cuda = torch.cuda.is_available()
    logging.info(f"GPU available: {use_cuda}")
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    best_epoch = 0
    args = parser.parse_args()
    # Set up logging file
    log_path = f"./result_log/{args.dataset}.log"
    setup_logging(log_path)

    args.cuda = torch.cuda.is_available()
    # logging.info(f"Using CUDA: {args.cuda}")
    args.device = torch.device("cuda" if args.cuda else "cpu")
    if args.dataset =='DHA':
        args.lr_train = 0.000001
        args.seed = 0
        args.main_view = 0
        args.con_epochs = 500
    elif args.dataset == 'Wiki_fea':
        args.lr_train = 0.00001
        args.seed =0
        args.main_view = 0
        args.con_epochs = 2000
    elif args.dataset == 'bbcsprot':
        args.lr_train = 0.002
        args.seed =0
        args.main_view = 0
    elif args.dataset == 'ALOI-100':
        args.lr_train = 0.000001
        args.seed = 0
        args.main_view = 0
    elif args.dataset == 'GSE100866':
        args.lr_train = 0.00001
        args.seed = 0
        args.main_view = 0

    set_seed(args.seed)
    dataloader, num_sample, dim_list, aligned_index, unaligned_index,y = MultiViewDatasetLoader(args)
    train_loader = DataLoader(dataloader,batch_size=args.batch_size,shuffle=True, drop_last=True)
    test_loader = DataLoader(dataloader,batch_size=num_sample,shuffle=False)
    args.num_cluster = np.size(np.unique(y[0]))
    print('num_cluster:',args.num_cluster)
    best_scores_kmeans = [0, 0, 0, 0, 0, 0, 0, 0]
    logging.info('unalign_ratio:{} lr:{} seed:{} main_view:{} alpha:{} beta:{}'.format(args.unalign_ratio, args.lr_train, args.seed,args.main_view,args.alpha,args.beta))
    network = Network(args,dim_list, device).to(device)
    optimizer = torch.optim.Adam(
        chain(network.parameters()),
        lr=args.lr_train,
        weight_decay=0.0000001
    )


    for epoch in range(args.con_epochs):
        loss_all = []
        rec_loss_all = []
        S_loss_all = []
        B_loss_all = []
        loss_fct = CrossEn()
        kl = KL()
        for batch_idx,(data0,data1,true_label) in enumerate(train_loader):
            data0 = data0.to(device)
            data1 = data1.to(device)
            re_data0, re_data1,z0,z1,M_t2v_logits, M_v2t_logits, logits,banzhaf,teacher= network(data0, data1)
            rec_loss = F.mse_loss(re_data0, data0) + F.mse_loss(re_data1, data1)
            rec_loss_all.append(rec_loss.item())
            S_loss_t2v = loss_fct(M_t2v_logits)
            S_loss_v2t = loss_fct(M_v2t_logits)
            S_loss = (S_loss_t2v + S_loss_v2t)/2
            s_loss = kl(banzhaf,teacher) + kl(banzhaf.T,teacher.T)
            S_loss_all.append((S_loss).item())
            B_loss_all.append((s_loss).item())
            loss = rec_loss +args.beta* S_loss + args.alpha*s_loss
            loss_all.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} loss: {np.mean(loss_all)} Rec_loss: {np.mean(rec_loss_all)} S_loss: {np.mean(S_loss_all)} B_loss:{np.mean(B_loss_all)} ")
        network.eval()
        with (torch.no_grad()):
            for batch_idx,(data0,data1,true_label) in enumerate(test_loader):
                data0 = data0.to(device)
                data1 = data1.to(device)
                z0 = network.encoders[0](data0)
                z1 = network.encoders[1](data1)
                if args.unalign_ratio !=0:
                    sim_marix = torch.mm(z0[unaligned_index],z1[unaligned_index].t())
                    contrib0 = torch.norm(z0[unaligned_index], dim=1, keepdim=True)
                    contrib1 = torch.norm(z1[unaligned_index], dim=1, keepdim=True)
                    interaction = sim_marix-contrib0-contrib1
                    if args.main_view ==0:
                        data_reranged = data0.clone()
                        for i in range(len(unaligned_index)):
                            idx = torch.argsort(interaction[:,i], descending=True)
                            data_reranged[unaligned_index[i]] = data0[unaligned_index][idx[0]]
                        for batch_idx,(data0,data1,true_label) in enumerate(test_loader):
                            data1 = data1.to(device)
                            z0 = network.encoders[0](data_reranged)
                            z1 = network.encoders[1](data1)
                    elif args.main_view == 1:
                        data_reranged = data1.clone()
                        for i in range(len(unaligned_index)):
                            idx = torch.argsort(interaction[:, i], descending=True)
                            data_reranged[unaligned_index[i]] = data1[unaligned_index][idx[0]]
                        for batch_idx, (data0, data1, true_label) in enumerate(test_loader):
                            data0 = data0.to(device)
                            z0 = network.encoders[0](data0)
                            z1 = network.encoders[1](data_reranged)
                y1 = torch.tensor(true_label, dtype=torch.int).to(device).detach().cpu().numpy()
                z_both = torch.cat((z0, z1), dim=1)
                kwargs = {
                    'metric': 'cosine',
                    'distributed': False,
                    'random_state': args.seed,
                    'n_clusters': args.num_cluster,
                    'verbose': False
                }
                km_torch = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
                psedo_labels = km_torch.fit_predict(z_both)
                ACC, NMI, ARI, Purity, Fscore, Precision, Recall, AMI = clusteringMetrics(y1,
                                                                                          psedo_labels.cpu().numpy())
                ACC = np.round(ACC, 4).item()
                NMI = np.round(NMI, 4).item()
                ARI = np.round(ARI, 4).item()
                Purity = np.round(Purity, 4).item()
                Fscore = np.round(Fscore, 4).item()
                Precision = np.round(Precision, 4).item()
                Recall = np.round(Recall, 4).item()
                AMI = np.round(AMI, 4).item()
                scores = [ACC, NMI, ARI, Purity, Fscore, Precision, Recall, AMI]
                ss = dict(
                    {'Epoch':epoch,'ACC': ACC, 'NMI': NMI, 'ARI': ARI, 'Purity': Purity, 'F-score': Fscore, 'Precision': Precision,
                     'Recall': Recall, 'AMI': AMI})
                print(ss)
                if scores[0]>best_scores_kmeans[0]:
                    best_scores_kmeans = scores
                    best_epoch = epoch
    logging.info(
        'Epoch: {} ACC = {} NMI = {}   ARI = {},Purity = {}, Fscore = {},Precision = {}, Recall = {}, AMI = {}'.format(
            best_epoch,
            best_scores_kmeans[0], best_scores_kmeans[1], best_scores_kmeans[2],
            best_scores_kmeans[3], best_scores_kmeans[4], best_scores_kmeans[5],
            best_scores_kmeans[6], best_scores_kmeans[7]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_z', default=128, type=int, help='choose from [32, 64]')
    parser.add_argument('--lr_train', default=0.0007, type=float, help='choose from [0.0001~0.001]')
    parser.add_argument('--batch_size', default=256, type=int, help='choose from [512, 1024, 2048]')  # fix
    # Data
    parser.add_argument('--dataset', default='DHA', type=str, help='choose dataset')
    parser.add_argument('--unalign_ratio', default=1, type=float,help='unalginment ratio')
    parser.add_argument('--main_view', default=1, type=int,   help='main view to obtain the final clustering assignments, from[0, 1]')
    # Train
    parser.add_argument('--con_epochs', type=int, default=200)
    parser.add_argument('--temper', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=24)
    parser.add_argument('--num_cluster',type=int,default=10)
    parser.add_argument('--alpha',type=float, default=0.1)
    parser.add_argument('--beta', type=float,default=0.1)
    args = parser.parse_args()
    main()