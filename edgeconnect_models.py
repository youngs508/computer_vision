import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# -------------------------
# Simple UNet Generators
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, base=64):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)

        self.mid = ConvBlock(base*4, base*8)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec3 = ConvBlock(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = ConvBlock(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec1 = ConvBlock(base*2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        m  = self.mid(self.pool3(e3))

        d3 = self.up3(m)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)

class EdgeGenerator(nn.Module):
    """
    Input: masked image (3) + mask (1) -> edge logits (1)
    """
    def __init__(self, base=64):
        super().__init__()
        self.net = UNet(in_ch=4, out_ch=1, base=base)

    def forward(self, masked_img, mask):
        x = torch.cat([masked_img, mask], dim=1)
        return self.net(x)  # logits

class InpaintGenerator(nn.Module):
    """
    Input: masked image (3) + edge (1) + mask (1) -> rgb logits (3)
    """
    def __init__(self, base=64):
        super().__init__()
        self.net = UNet(in_ch=5, out_ch=3, base=base)

    def forward(self, masked_img, edge, mask):
        x = torch.cat([masked_img, edge, mask], dim=1)
        return self.net(x)  # logits

# -------------------------
# PatchGAN Discriminator
# -------------------------
class PatchDiscriminator(nn.Module):
    """
    Returns:
      logits: [B,1,H',W']
      feats: list of intermediate feature maps (for feature matching)
    """
    def __init__(self, in_ch, base=64, n_layers=4):
        super().__init__()
        layers = []
        feats_out = []

        ch = base
        layers.append(nn.Conv2d(in_ch, ch, 4, 2, 1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        for i in range(1, n_layers):
            ch_prev = ch
            ch = min(ch * 2, 512)
            layers.append(nn.Conv2d(ch_prev, ch, 4, 2, 1))
            layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # one more conv (stride=1) to refine
        layers.append(nn.Conv2d(ch, ch, 4, 1, 1))
        layers.append(nn.BatchNorm2d(ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # output logits
        layers.append(nn.Conv2d(ch, 1, 4, 1, 1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        feats = []
        h = x
        for layer in self.layers[:-1]:
            h = layer(h)
            if isinstance(layer, nn.LeakyReLU):
                feats.append(h)
        logits = self.layers[-1](h)
        return logits, feats

# -------------------------
# VGG Perceptual (features)
# -------------------------
class VGG16FeatureExtractor(nn.Module):
    """
    Perceptual loss uses fixed VGG16 features.
    Layers commonly used: relu1_2, relu2_2, relu3_3
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*[vgg[x] for x in range(0, 4)])   # relu1_2
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(4, 9)])   # relu2_2
        self.slice3 = nn.Sequential(*[vgg[x] for x in range(9, 16)])  # relu3_3
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        # x expected in [0,1], normalize to ImageNet stats
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std  = x.new_tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        x = (x - mean) / std

        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        return [h1, h2, h3]

# -------------------------
# Loss helpers
# -------------------------
def gan_loss_logits(logits, is_real: bool):
    # LSGAN (more stable than BCE GAN)
    target = torch.ones_like(logits) if is_real else torch.zeros_like(logits)
    return F.mse_loss(logits, target)

def feature_matching_loss(fake_feats, real_feats):
    loss = 0.0
    for f, r in zip(fake_feats, real_feats):
        loss = loss + F.l1_loss(f, r.detach())
    return loss
