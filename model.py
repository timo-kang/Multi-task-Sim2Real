import torch.nn as nn
import torch
import numpy as np


class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.hidden_dim = 128
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(7, 1024)


        self.lstm = nn.LSTM(input_size=1024, hidden_size=self.hidden_dim, batch_first=True)
        # Linear model (width*height*channel of the last feature map, Number of class)
        self.last = nn.Linear(self.hidden_dim, 15)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 64, self.hidden_dim).cuda(),
                torch.zeros(1, 64, self.hidden_dim).cuda())

    def forward(self, x): # [B, 4W, H, C]

        flatten = []
        for i in range(4):
            # [B, 3, 256, 256] -> [B, 256, 2, 2]
            out = self.conv_layer(x[:, :, i*x.size(3):(i+1)*x.size(3), :])
            flatten.append(out.reshape(x.size(0), 1, -1))  # Flatten [B, 1, 1024]

        flattens = torch.cat(flatten, dim=1)  # [B, 4, 1024]
        lstm_out, self.hidden = self.lstm(flattens, self.init_hidden()) # [B, SL, IS]
        lstm_out = lstm_out[:, -1, :]
        out = self.last(lstm_out) # [B, 1, H]
        return out
