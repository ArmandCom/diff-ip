import argparse
import random
import time
import glob
from tqdm import tqdm   
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from arch.mnist import ClassifierMNIST, QuerierMNIST
import ops
import utils
import wandb



def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_queries', type=int, default=676)
    parser.add_argument('--max_queries_test', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--tau_start', type=float, default=1.0)
    parser.add_argument('--tau_end', type=float, default=0.2)
    parser.add_argument('--sampling', type=str, default='random')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='mnist')
    parser.add_argument('--mode', type=str, default='online')
    parser.add_argument('--tail', type=str, default='', help='tail message')
    parser.add_argument('--ckpt_class', type=str, default=None, help='load checkpoint')
    parser.add_argument('--ckpt_quer', type=str, default=None, help='load checkpoint')
    parser.add_argument('--save_dir', type=str, default='./saved/', help='save directory')
    parser.add_argument('--run_name', type=str, default='mnist_class_mix', help='save directory')
    parser.add_argument('--data_dir', type=str, default='./data/', help='save directory')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_id', type=str, default="1")


    args = parser.parse_args()
    return args

def main(args):
    ## Setup
    # wandb
    run = wandb.init(project="Variational-IP", name=args.name, mode=args.mode)
    # model_dir = os.path.join(args.save_dir, f'{run.id}')
    model_dir = os.path.join(args.save_dir, f'{run.id}')
    os.makedirs(model_dir, exist_ok=True)

    model_dir_ckpt = os.path.join(args.save_dir, args.run_name)
    os.makedirs(os.path.join(model_dir_ckpt, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, args.run_name), exist_ok=True)

    utils.save_params(model_dir_ckpt, vars(args)) #TODO: or model_dir_ckpt?
    wandb.config.update(args)

    # cuda
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('DEVICE:', device)
    print( 'N Dev, Curr Dev', torch.cuda.device_count(), torch.cuda.current_device())

    # random
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ## Constants
    QUERY_ALL = 676 # 26*26
    PATCH_SIZE = 3
    THRESHOLD = 0.85

    ## Data
    transform = transforms.Compose([transforms.ToTensor(),  
                                    transforms.Lambda(lambda x: torch.where(x < 0.5, -1., 1.))])
    trainset = datasets.MNIST(args.data_dir, train=True, transform=transform, download=True)
    testset = datasets.MNIST(args.data_dir, train=False, transform=transform, download=True)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers)

    ## Model
    classifier = ClassifierMNIST()
    classifier = nn.DataParallel(classifier).to(device)
    querier = QuerierMNIST(num_classes=QUERY_ALL, tau=args.tau_start)
    querier = nn.DataParallel(querier).to(device)

    ## Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(querier.parameters()) + list(classifier.parameters()), 
                           amsgrad=True, lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    tau_vals = np.linspace(args.tau_start, args.tau_end, args.epochs)

    ## Load checkpoint
    ckpt_dict_class = torch.load(args.ckpt_class, map_location='cpu')
    ckpt_dict_quer = torch.load(args.ckpt_quer, map_location='cpu')
    classifier.load_state_dict(ckpt_dict_class['classifier'])
    querier.load_state_dict(ckpt_dict_quer['querier'])
    querier.load_state_dict(ckpt_dict_class['querier'])
    optimizer.load_state_dict(ckpt_dict_class['optimizer'])
    scheduler.load_state_dict(ckpt_dict_class['scheduler'])
    print('Checkpoint Loaded!')

    ## Train
    for epoch in range(args.epochs):

        # training
        # classifier.train()
        # querier.train()
        # tau = tau_vals[epoch]
        # for train_images, train_labels in tqdm(trainloader):
        #     train_images = train_images.to(device)
        #     train_labels = train_labels.to(device)
        #     querier.module.update_tau(tau)
        #     optimizer.zero_grad()
        #
        #     # initial random sampling
        #     if args.sampling == 'biased':
        #         num_queries = torch.randint(low=0, high=QUERY_ALL, size=(train_images.size(0),))
        #         mask, masked_image = ops.adaptive_sampling(train_images, num_queries, querier, PATCH_SIZE, QUERY_ALL)
        #     elif args.sampling == 'random':
        #         mask = ops.random_sampling(args.max_queries, QUERY_ALL, train_images.size(0)).to(device)
        #         masked_image = ops.get_patch_mask(mask, train_images, patch_size=PATCH_SIZE)
        #
        #     # Query and update
        #     query_vec = querier(masked_image, mask)
        #     masked_image = ops.update_masked_image(masked_image, train_images, query_vec, patch_size=PATCH_SIZE)
        #
        #     # prediction
        #     train_logits = classifier(masked_image)
        #
        #     # backprop
        #     loss = criterion(train_logits, train_labels)
        #     loss.backward()
        #     optimizer.step()
        #
        #     # logging
        #     wandb.log({
        #         'epoch': epoch,
        #         'loss': loss.item(),
        #         'lr': utils.get_lr(optimizer),
        #         'gradnorm_cls': utils.get_grad_norm(classifier),
        #         'gradnorm_qry': utils.get_grad_norm(querier)
        #         })
        # scheduler.step()
        #
        # # saving
        # if epoch % 10 == 0 or epoch == args.epochs - 1:
        #     torch.save({
        #         'classifier': classifier.state_dict(),
        #         'querier': querier.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'scheduler': scheduler.state_dict()
        #         },
        #         os.path.join(model_dir_ckpt, 'ckpt', f'epoch{epoch}.ckpt'))

        # evaluation
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            classifier.eval()
            querier.eval()
            epoch_test_qry_need = []
            epoch_test_acc_max = 0
            epoch_test_acc_ip = 0
            for test_images, test_labels in tqdm(testloader):
                test_images = test_images.to(device)
                test_labels = test_labels.to(device)
                N, H, C, W = test_images.shape

                # Compute logits for all queries
                test_inputs = torch.zeros_like(test_images).to(device)
                mask = torch.zeros(N, QUERY_ALL).to(device)
                logits, queries = [], []
                for i in range(args.max_queries_test):
                    with torch.no_grad():
                        query_vec = querier(test_inputs, mask)
                        label_logits = classifier(test_inputs)

                    mask[np.arange(N), query_vec.argmax(dim=1)] = 1.0
                    test_inputs = ops.update_masked_image(test_inputs, test_images, query_vec, patch_size=PATCH_SIZE)
                    
                    logits.append(label_logits)
                    queries.append(query_vec)
                logits = torch.stack(logits).permute(1, 0, 2)

                # accuracy using all queries
                test_pred_max = logits[:, -1, :].argmax(dim=1).float()
                test_acc_max = (test_pred_max == test_labels.squeeze()).float().sum()
                epoch_test_acc_max += test_acc_max

                # compute query needed
                qry_need = ops.compute_queries_needed(logits, threshold=THRESHOLD)
                epoch_test_qry_need.append(qry_need)

                # accuracy using IP
                test_pred_ip = logits[torch.arange(len(qry_need)), qry_need-1].argmax(1)
                test_acc_ip = (test_pred_ip == test_labels.squeeze()).float().sum()
                epoch_test_acc_ip += test_acc_ip
            epoch_test_acc_max = epoch_test_acc_max / len(testset)
            epoch_test_acc_ip = epoch_test_acc_ip / len(testset)

            # mean and std of queries needed
            epoch_test_qry_need = torch.hstack(epoch_test_qry_need).float()
            qry_need_avg = epoch_test_qry_need.mean()
            qry_need_std = epoch_test_qry_need.std()

            dict_log = {
                'test_epoch': epoch,
                'test_acc_max': epoch_test_acc_max,
                'test_acc_ip': epoch_test_acc_ip,
                'qry_need_avg': qry_need_avg,
                'qry_need_std': qry_need_std
            }
            for k,v in dict_log.items():
                print(k + ': ' + str(v))
            # logging
            wandb.log(dict_log)
            exit()


if __name__ == '__main__':
    args = parseargs()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

    main(args)


