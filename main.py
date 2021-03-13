import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import argparse
import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

from models import ConvDiscriminator, ConvGenerator, LSTMDiscriminator, LSTMGenerator
from utils import make_dirs, pre_processing, post_processing, prepare_data, moving_windows, get_lr_scheduler
from utils import get_gradient_penalty, plot_series, get_samples, generate_fake_samples, make_csv, derive_metrics

# Reproducibility #
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(args):

    # Device Configuration #
    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')

    # Fix Seed for Reproducibility #
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Samples, Plots, Weights and CSV Path #
    paths = [args.samples_path, args.weights_path, args.csv_path, args.inference_path]
    for path in paths:
        make_dirs(path)

    # Prepare Data #
    data = pd.read_csv(args.data_path)[args.column]

    # Prepare Data #
    scaler_1 = StandardScaler()
    scaler_2 = StandardScaler()
    preprocessed_data = pre_processing(data, scaler_1, scaler_2, args.constant, args.delta)

    train_X, train_Y, test_X, test_Y = prepare_data(data, preprocessed_data, args)

    train_X = moving_windows(train_X, args.ts_dim)
    train_Y = moving_windows(train_Y, args.ts_dim)

    test_X = moving_windows(test_X, args.ts_dim)
    test_Y = moving_windows(test_Y, args.ts_dim)

    # Prepare Networks #
    if args.model == 'conv':
        D = ConvDiscriminator(args.ts_dim).to(device)
        G = ConvGenerator(args.latent_dim, args.ts_dim).to(device)
    
    elif args.model == 'lstm':
        D = LSTMDiscriminator(args.ts_dim).to(device)
        G = LSTMGenerator(args.latent_dim, args.ts_dim).to(device)
    
    else:
        raise NotImplementedError        
    
    #########
    # Train #
    #########

    if args.mode == 'train':
        
        # Loss Function #
        if args.criterion == 'l2':
            criterion = nn.MSELoss()
        
        elif args.criterion == 'wgangp':
            pass
        
        else:
            raise NotImplementedError

        # Optimizers #
        if args.optim == 'sgd':
            D_optim = torch.optim.SGD(D.parameters(), lr=args.lr, momentum=0.9)
            G_optim = torch.optim.SGD(G.parameters(), lr=args.lr, momentum=0.9)
        
        elif args.optim == 'adam':
            D_optim = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0., 0.9))
            G_optim = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0., 0.9))
        
        else:
            raise NotImplementedError

        D_optim_scheduler = get_lr_scheduler(D_optim, args)
        G_optim_scheduler = get_lr_scheduler(G_optim, args)

        # Lists #
        D_losses, G_losses = list(), list()

        # Train #
        print("Training Time Series GAN started with total epoch of {}.".format(args.num_epochs))
        
        for epoch in range(args.num_epochs):

            # Initialize Optimizers #
            G_optim.zero_grad()
            D_optim.zero_grad()

            #######################
            # Train Discriminator #
            #######################

            if args.criterion == 'l2':
                n_critics = 1
            elif args.criterion == 'wgangp':
                n_critics = 5

            for j in range(n_critics):
                series, start_dates = get_samples(train_X, train_Y, args.batch_size)

                # Data Preparation #
                series = series.to(device)
                noise = torch.randn(args.batch_size, 1, args.latent_dim).to(device)

                # Adversarial Loss using Real Image #
                prob_real = D(series.float())
                
                if args.criterion == 'l2':
                    real_labels = torch.ones(prob_real.size()).to(device)
                    D_real_loss = criterion(prob_real, real_labels)

                elif args.criterion == 'wgangp':
                    D_real_loss = -torch.mean(prob_real)

                # Adversarial Loss using Fake Image #
                fake_series = G(noise)
                prob_fake = D(fake_series.detach())

                if args.criterion == 'l2':
                    fake_labels = torch.zeros(prob_fake.size()).to(device)
                    D_fake_loss = criterion(prob_fake, fake_labels)

                elif args.criterion == 'wgangp':
                    D_fake_loss = torch.mean(prob_fake)
                    D_gp_loss = args.lambda_gp * get_gradient_penalty(D, series.float(), fake_series.float(), device)

                # Calculate Total Discriminator Loss #
                D_loss = D_fake_loss + D_real_loss
                    
                if args.criterion == 'wgangp':
                    D_loss += args.lambda_gp * D_gp_loss
                            
                # Back Propagation and Update #
                D_loss.backward()
                D_optim.step()

            ###################
            # Train Generator #
            ###################

            # Adversarial Loss #
            fake_series = G(noise)
            prob_fake = D(fake_series)

            # Calculate Total Generator Loss #
            if args.criterion == 'l2':
                real_labels = torch.ones(prob_fake.size()).to(device)
                G_loss = criterion(prob_fake, real_labels)

            elif args.criterion == 'wgangp':
                G_loss = -torch.mean(prob_fake)

            # Back Propagation and Update #
            G_loss.backward()
            G_optim.step()

            # Add items to Lists #
            D_losses.append(D_loss.item())
            G_losses.append(G_loss.item())

            # Adjust Learning Rate #
            D_optim_scheduler.step()
            G_optim_scheduler.step()

            # Print Statistics, Save Model Weights and Series #
            if (epoch+1) % args.log_every == 0:            
                
                # Print Statistics and Save Model #
                print("Epochs [{}/{}] | D Loss {:.4f} | G Loss {:.4f}".format(epoch+1, args.num_epochs, np.average(D_losses), np.average(G_losses)))
                torch.save(G.state_dict(), os.path.join(args.weights_path, 'TS_using{}_and_{}_Epoch_{}.pkl'.format(G.__class__.__name__, args.criterion.upper(), epoch + 1)))

                # Generate Samples and Save Plots and CSVs #
                series, fake_series = generate_fake_samples(test_X, test_Y, G, scaler_1, scaler_2, args, device)
                plot_series(series, fake_series, G, epoch, args, args.samples_path)
                make_csv(series, fake_series, G, epoch, args, args.csv_path)
    
    ########
    # Test #
    ########

    elif args.mode == 'test':
        
        # Load Model Weights #
        G.load_state_dict(torch.load(os.path.join(args.weights_path, 'TS_using{}_and_{}_Epoch_{}.pkl'.format(G.__class__.__name__, args.criterion.upper(), args.num_epochs))))


        # Lists #
        real, fake = list(), list()

        # Inference #
        for idx in range(0, test_X.shape[0], args.ts_dim):
            
            # Do not plot if the remaining data is less than time dimension #
            end_ix = idx + args.ts_dim

            if end_ix > len(test_X)-1:
                break
            
            # Prepare Data #
            test_data = test_X[idx, :]
            test_data = np.expand_dims(test_data, axis=0)
            test_data = np.expand_dims(test_data, axis=1)
            test_data = torch.from_numpy(test_data).to(device)

            start = test_Y[idx, 0]

            noise = torch.randn(args.val_batch_size, 1, args.latent_dim).to(device)

            # Generate Fake Data #
            with torch.no_grad():
                fake_series = G(noise)

            # Convert to Numpy format for Saving #
            test_data = np.squeeze(test_data.cpu().data.numpy())
            fake_series = np.squeeze(fake_series.cpu().data.numpy())
            
            test_data = post_processing(test_data, start, scaler_1, scaler_2, args.delta)
            fake_series = post_processing(fake_series, start, scaler_1, scaler_2, args.delta)

            real += test_data.tolist()
            fake += fake_series.tolist()
        
        # Plot, Save to CSV file and Derive Metrics #
        plot_series(real, fake, G, args.num_epochs-1, args, args.inference_path)
        make_csv(real, fake, G, args.num_epochs-1, args, args.inference_path)
        derive_metrics(real, fake, args)
        
    else:
        raise NotImplementedError

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_num', type=int, default=5, help='gpu number')
    parser.add_argument('--seed', type=int, default=7777, help='seed')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train or test')

    parser.add_argument('--data_path', type=str, default='./data/energydata_complete.csv', help='data path')
    parser.add_argument('--column', type=str, default='Appliances', help='which column to generate')
    parser.add_argument('--train_split', type=float, default=0.8, help='train-test split ratio')

    parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size')
    parser.add_argument('--val_batch_size', type=int, default=1, help='mini-batch size for validation')
    parser.add_argument('--num_epochs', type=int, default=1000, help='total epoch for training')
    parser.add_argument('--log_every', type=int, default=50, help='save log data for every default iteration')
    parser.add_argument('--metric_iteration', type=int, default=5, help='iterate calculation for metrics for evaluation')

    parser.add_argument('--model', type=str, default='conv', choices=['conv', 'lstm'], help='which network to train')
    parser.add_argument('--delta', type=float, default=0.7, help='delta')
    parser.add_argument('--constant', type=float, default=0.0, help='If zero in the original data, please set it as non-zero, e.g. 1e-1')
    parser.add_argument('--ts_dim', type=int, default=100, help='time series dimension, how many time steps to synthesize')
    parser.add_argument('--latent_dim', type=int, default=25, help='noise dimension')

    parser.add_argument('--criterion', type=str, default='wgangp', choices=['l2', 'wgangp'], help='criterion')
    parser.add_argument('--lambda_gp', type=int, default=10, help='constant for gradient penalty')

    parser.add_argument('--optim', type=str, default='adam', choices=['sgd', 'adam'], help='which optimizer to update')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='decay learning rate')
    parser.add_argument('--lr_decay_every', type=int, default=1000, help='decay learning rate for every default epoch')
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['step', 'plateau', 'cosine'], help='learning rate scheduler')

    parser.add_argument('--samples_path', type=str, default='./results/samples/', help='samples path')
    parser.add_argument('--weights_path', type=str, default='./results/weights/', help='weights path')
    parser.add_argument('--csv_path', type=str, default='./results/csv/', help='csv path')
    parser.add_argument('--inference_path', type=str, default='./results/inference/', help='inference path')

    args = parser.parse_args()

    torch.cuda.empty_cache()
    main(args)