import argparse
import datetime
import os
import random
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

# from dataset import MovingMNIST
from dataset import MovingMNIST
from model import TDVAE


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Temporal Difference Variational Auto-Encoder Implementation')
    parser.add_argument('--epochs', type=int, help='number of epochs (default: 1000)', default=1000)
    parser.add_argument('--batch_size', type=int, default=100, help='size of batch (default: 100)')
    parser.add_argument('--dataset_type', type=str, help='type of dataset', default='MovingMNIST')
    parser.add_argument('--root_log_dir', type=str, help='root location of log', default='/workspace/logs/')
    parser.add_argument('--log_dir', type=str, help='log directory', default='HierarchicalTDVAE')
    parser.add_argument('--log_interval', type=int, help='interval number of steps for logging', default=200)
    parser.add_argument('--save_interval', type=int, help='interval number of steps for saveing models', default=1000)
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--seed', type=int, help='random seed (default: None)', default=1234)
    parser.add_argument('--device_ids', type=int, nargs='+', help='list of CUDA devices (default: [0])', default=[0])
    parser.add_argument('--z_size', type=int, help='size of latent space(z)', default=8)
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--rescale', type=float, help='resize scale of input', default=None)
    args = parser.parse_args()
    
    # Device
    device = f"cuda:{args.device_ids[0]}" if torch.cuda.is_available() else "cpu"
    
    # Seed
    if args.seed!=None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    
    # Logging
    log_interval_num = args.log_interval
    log_dir = os.path.join(args.root_log_dir, args.log_dir)
    os.mkdir(log_dir)
    os.mkdir(os.path.join(log_dir, 'models'))
    os.mkdir(os.path.join(log_dir,'runs'))
    writer = SummaryWriter(log_dir=os.path.join(log_dir,'runs'))

    train_dataset = MovingMNIST('/workspace/dataset/MNIST')
    test_dataset = MovingMNIST('/workspace/dataset/MNIST', train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    train_loader_iterator = iter(train_loader)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_batch = next(iter(test_loader)).to(device)
    
    model = TDVAE().to(device)
    if len(args.device_ids)>1:
        model = nn.DataParallel(model, device_ids=args.device_ids)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    itr = 0
    for epoch in range(args.epochs):
        for batch in tqdm(train_loader):
            batch = batch.to(device)
            loss = model(batch).mean()
            loss.backward()
            writer.add_scalar('train_loss', loss, itr)
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                if itr % log_interval_num == 0:
                    if len(args.device_ids)>1:
                        kl, entropy, ce, nll, test_pred = model.module.test(test_batch)
                    else:
                        kl, entropy, ce, nll, test_pred = model.test(test_batch)
                    test_loss = (kl + entropy + ce + nll).mean()
                    writer.add_scalar('test_loss', test_loss, itr)
                    writer.add_scalar('test_kl', kl.mean(), itr)
                    writer.add_scalar('test_entropy', entropy.mean(), itr)
                    writer.add_scalar('test_ce', ce.mean(), itr)
                    writer.add_scalar('test_nll', nll.mean(), itr)
                    writer.add_video('test_pred', test_pred[:64], itr)
                    writer.add_video('test_ground_truth', test_batch[:64], itr)
            
            itr += 1

    writer.close()
