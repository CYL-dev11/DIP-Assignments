import torch
import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        
        # --- Encoder (下采样: 逐步减小空间尺寸，增加通道数) ---
        # 输入: [3, 256, 256]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # -> [8, 128, 128]
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True) # Pix2Pix 编码器通常使用 LeakyReLU
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1), # -> [16, 64, 64]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),# -> [32, 32, 32]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),# -> [64, 16, 16]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# -> [128, 8, 8]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# -> [256, 4, 4]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # --- Decoder (上采样: 逐步恢复空间尺寸，减少通道数) ---
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> [128, 8, 8]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> [64, 16, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> [32, 32, 32]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # -> [16, 64, 64]
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),    # -> [8, 128, 128]
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        # 最后输出层，将通道数恢复为 3
        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),     # -> [3, 256, 256]
            nn.Tanh() 
        )

    def forward(self, x):
        # Encoder forward pass
        e1 = self.conv1(x)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        e6 = self.conv6(e5)

        # Decoder forward pass
        d1 = self.deconv1(e6)
        d2 = self.deconv2(d1)
        d3 = self.deconv3(d2)
        d4 = self.deconv4(d3)
        d5 = self.deconv5(d4)
        output = self.deconv6(d5)
        
        return output
