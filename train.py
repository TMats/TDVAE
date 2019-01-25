import argparse
import datetime
import os
import random
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MovingMNIST, MovingMNISTLR
from model import TDVAE


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Temporal Difference Variational Auto-Encoder Implementation')
    parser.add_argument('--gradient_steps', type=int, default=2*10**4, help='number of gradient steps to run')
    parser.add_argument('--batch_size', type=int, default=32, help='size of batch (default: 32)')
    parser.add_argument('--dataset_type', type=str, help='type of dataset', default='MovingMNIST')
    parser.add_argument('--root_log_dir', type=str, help='root location of log', default='../log/TDVAE/')
    parser.add_argument('--data_dir', type=str, help='root location of dataset', default='data')
    parser.add_argument('--log_dir', type=str, help='log directory', default=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
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
    
    # Dataset
    if args.dataset_type == 'MovingMNIST':
        data_path = os.path.join(args.data_dir, 'mnist_test_seq.npy')
        full_dataset = MovingMNIST(data_path, rescale=args.rescale)
        data_num =  len(full_dataset)
        train_size = int(0.9 * data_num)
        test_size = data_num - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    elif args.dataset_type == 'MovingMNISTLR':
        train_dataset = MovingMNISTLR(args.data_dir, train=True, download=True)
        test_dataset = MovingMNISTLR(args.data_dir, train=False, download=True)
    else:
        raise NotImplementedError()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    train_loader_iterator = iter(train_loader)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader_iterator = iter(test_loader)
    test_batch = next(test_loader_iterator).to(device)
    _, _, C, H, W = test_batch.size()

    model = TDVAE(z_size=args.z_size, x_size=C*H*W, processed_x_size=C*H*W).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for itr in tqdm(range(args.gradient_steps)):
        try:
            batch = next(train_loader_iterator)
        except StopIteration:
            train_loader_iterator = iter(train_loader)
            batch = next(train_loader_iterator)
        batch = batch.to(device)
        
        loss = model(batch)
        loss.backward()
        writer.add_scalar('train_loss', loss, itr)
        optimizer.step()
        optimizer.zero_grad()
        
        with torch.no_grad():
            if itr % log_interval_num == 0:
                test_pred, kl, b_ll, t_nll, d_nll = model.test(test_batch)
                test_loss = kl + b_ll + t_nll +d_nll
                writer.add_scalar('test_loss', test_loss, itr)
                writer.add_scalar('test_kl', kl, itr)
                writer.add_scalar('test_b_ll', b_ll, itr)
                writer.add_scalar('test_t_nll', t_nll, itr)
                writer.add_scalar('test_d_nll', d_nll, itr)
                writer.add_video('test_pred', test_pred, itr)
                writer.add_video('test_ground_truth', test_batch, itr)

    writer.close()
