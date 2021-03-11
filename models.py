import torch
import torch.nn as nn


class ConvDiscriminator(nn.Module):
    """Discriminator with Dilated Convolution"""
    def __init__(self, ts_dim, hidden_dim=512, out_dim=1):
        super(ConvDiscriminator, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(ts_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
        )
        
        self.linear_block = nn.Sequential(    
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):

        out = self.feature(x)
        
        block = self.linear_block(out)
        out2 = out + block
        
        block2 = self.linear_block(out2)
        out3 = out2 + block2
        
        block3 = self.linear_block(out3)
        out4 = out3 + block3
        
        block4 = self.linear_block(out4)
        out5 = out4 + block4
        
        block5 = self.linear_block(out5)
        out6 = out5 + block5
        
        block6 = self.linear_block(out6)
        out7 = out6 + block6
        
        block7 = self.linear_block(out7)
        out8 = out7 + block7

        out = self.fc(out8)
        
        return out


class ConvGenerator(nn.Module):
    """Generator with Dilated Convolution"""
    def __init__(self, latent_dim, ts_dim, hidden_dim=256, hidden_dim2=10):
        super(ConvGenerator, self).__init__()
        
        self.latent = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=1, bias=False),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, dilation=2, padding=4, bias=False),
            nn.LeakyReLU(inplace=True),
        )

        self.linear_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(inplace=True)
        )

        self.conv1d_block = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2, padding=2, bias=False),
            nn.LeakyReLU(inplace=True),
        )

        self.shifting_block = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim2, kernel_size=3, dilation=2, padding=2, bias=False),
            nn.LeakyReLU(inplace=True),
            
            nn.Flatten(start_dim=1),

            nn.Linear(hidden_dim2 * latent_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, ts_dim),
        )

    def forward(self, x):
        out = self.latent(x)

        cnn_block = self.conv1d_block(out)
        out2 = cnn_block + out
        
        cnn_block2 = self.conv1d_block(out2)
        out3 = cnn_block2 + out2
        
        cnn_block3 = self.conv1d_block(out3)
        out4 = cnn_block3 + out3
        
        out4 = self.shifting_block(out4)
        
        block = self.linear_block(out4)
        out5 = block + out4
        
        block2 = self.linear_block(out5)
        out6 = block2 + out5
        
        block3 = self.linear_block(out6)
        out7 = block3 + out6
        
        out = self.fc(out7)        
        return out[:, None, :]


class LSTMDiscriminator(nn.Module):
    """Discriminator with LSTM"""
    def __init__(self, ts_dim, hidden_dim=256, num_layers=1):
        super(LSTMDiscriminator, self).__init__()
        
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(ts_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out.view(x.size(0) * x.size(1), self.hidden_dim))
        out = out.view(x.size(0), x.size(1), 1)
        return out


class LSTMGenerator(nn.Module):
    """Generator with LSTM"""
    def __init__(self, latent_dim, ts_dim, hidden_dim=256, num_layers=1):
        super(LSTMGenerator, self).__init__()

        self.ts_dim = ts_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, ts_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out.view(x.size(0) * x.size(1), self.hidden_dim))
        out = out.view(x.size(0), x.size(1), self.ts_dim)
        return out