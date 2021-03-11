import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.special import lambertw

import torch
from torch.autograd import grad

from metrics import real_data_loading, discriminative_score_metrics, predictive_score_metrics


def make_dirs(path):
    """Make Directory If not Exists"""
    if not os.path.exists(path):
        os.makedirs(path)


def pre_processing(data, scaler_1, scaler_2, constant, delta):
    """Pre-processing"""
    data += constant
    log_returns = np.log(data/data.shift(1)).fillna(0).to_numpy()
    log_returns = np.reshape(log_returns, (log_returns.shape[0], 1))
    log_returns = scaler_1.fit_transform(log_returns)
    log_returns = np.squeeze(log_returns)
    
    log_returns_w = (np.sign(log_returns) * np.sqrt(lambertw(delta*log_returns**2)/delta)).real
    log_returns_w = log_returns_w.reshape(-1, 1)
    log_returns_w = scaler_2.fit_transform(log_returns_w)
    log_returns_w = np.squeeze(log_returns_w)

    return log_returns_w


def post_processing(data, start, scaler_1, scaler_2, delta):
    """Post-processing"""
    data = scaler_2.inverse_transform(data)
    data = data * np.exp(0.5 * delta * data **2)
    data = scaler_1.inverse_transform(data)
    data = np.exp(data)
    
    post_data = np.empty((data.shape[0], ))
    post_data[0] = start
    for i in range(1, data.shape[0]):
        post_data[i] = post_data[i-1] * data[i]

    return np.array(post_data)


def moving_windows(x, length):
    """Moving Windows"""
    windows = list()

    for i in range(0, len(x)):
        end_ix = i + length

        if end_ix > len(x)-1:
            break

        windows.append(x[i : i + length])

    return np.array(windows)

def prepare_data(data, preprocessed_data, args):
    """Prepare Data"""
    data = data.to_numpy()

    train_size = int(len(data) * args.train_split)
    test_size = len(data) - train_size

    train_X = preprocessed_data[0:train_size]
    train_Y = data[0:train_size]

    test_X = preprocessed_data[train_size:len(preprocessed_data)]
    test_Y = data[train_size:len(preprocessed_data)]

    return train_X, train_Y, test_X, test_Y


def get_lr_scheduler(optimizer, args):
    """Learning Rate Scheduler"""
    if args.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay_rate)
    elif args.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay_every, threshold=0.001, patience=1)
    elif args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax, eta_min=0)
    else:
        raise NotImplementedError

    return scheduler


def get_gradient_penalty(discriminator, real_images, fake_images, device, eps = 1e-12):
    """Gradient Penalty"""
    epsilon = torch.rand(real_images.size(0), 1, 1).to(device)
    epsilon = epsilon

    x_hat = (epsilon * real_images + (1 - epsilon) * fake_images).requires_grad_(True)
    x_hat_prob = discriminator(x_hat)
    x_hat_grad = torch.ones(x_hat_prob.size()).to(device)

    gradients = grad(outputs=x_hat_prob,
                     inputs=x_hat,
                     grad_outputs=x_hat_grad,
                     create_graph=True,
                     retain_graph=True,
                     only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = torch.sqrt(torch.sum(gradients ** 2, dim=1) + eps)
    gradient_penalty = torch.mean((gradient_penalty-1)**2)

    return gradient_penalty


def get_samples(data, label, batch_size):
    """Get Samples"""
    idx = np.random.randint(data.shape[0], size=batch_size)

    samples = data[idx, :]
    samples = np.expand_dims(samples, axis=1)
    samples = torch.from_numpy(samples)
    
    start = label[idx, 0]

    return samples, start


def generate_fake_samples(data, label, generator, scaler_1, scaler_2, args, device):
    """Generate Fake Samples"""
    series, start = get_samples(data, label, args.val_batch_size)
    series = series.to(device)

    noise = torch.randn(args.val_batch_size, 1, args.latent_dim).to(device)

    fake_series = generator(noise.detach())

    series = np.squeeze(series.cpu().data.numpy())
    fake_series = np.squeeze(fake_series.cpu().data.numpy())
        
    series = post_processing(series, start, scaler_1, scaler_2, args.delta)
    fake_series = post_processing(fake_series, start, scaler_1, scaler_2, args.delta)

    return series, fake_series


def plot_series(series, fake_series, generator, epoch, args, path):
    """Plot Samples"""
    plt.figure(figsize=(10, 5))
    plt.plot(series, label='real')
    plt.plot(fake_series, label='fake')
    plt.grid(True)
    plt.legend()
    plt.xlabel('time')
    plt.ylabel(args.column)
    plt.title('FakeTS_using{}_and_{}_Epoch{}.png'.format(generator.__class__.__name__, args.criterion.upper(), epoch+1))

    plt.savefig(os.path.join(path, 'FakeTS_using{}_and_{}_Epoch{}.png'.format(generator.__class__.__name__, args.criterion.upper(), epoch+1)))


def make_csv(series, fake_series, generator, epoch, args, path):
    """Convert to CSV files"""
    data = pd.DataFrame({'series' : series, 'fake_series' : fake_series})

    data.to_csv(
        os.path.join(path, 'FakeTS_using{}_and_{}_Epoch{}.csv'.format(generator.__class__.__name__, args.criterion.upper(), epoch+1)),
        header=['Real', 'Fake'],
        index=False
        )


def derive_metrics(ori_data, gen_data, args):
  """Derivation of metrics"""
  ori_data = real_data_loading(ori_data, args.ts_dim)
  gen_data = real_data_loading(gen_data, args.ts_dim)

  ori_data = np.expand_dims(ori_data, axis=2)
  gen_data = np.expand_dims(gen_data, axis=2)

  disc_score = discriminative_score_metrics(ori_data, gen_data)
  pred_score = predictive_score_metrics(ori_data, gen_data)

  print('Disc score: ' + str(disc_score))
  print('Pred score: ' + str(pred_score))