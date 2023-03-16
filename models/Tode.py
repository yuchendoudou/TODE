import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin_transformer import SwinTransformer
import math



class UpSampleBN(nn.Module):
    def __init__(self, input_features, output_features, res = True):
        super(UpSampleBN, self).__init__()
        self.res = res

        self._net = nn.Sequential(nn.Conv2d(input_features, input_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(input_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(input_features, input_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(input_features),
                                  nn.LeakyReLU())

        self.up_net = nn.Sequential(nn.ConvTranspose2d(input_features, output_features, kernel_size = 2, stride = 2, padding = 0, output_padding = 0),
                                    nn.BatchNorm2d(output_features, output_features),
                                    nn.ReLU(True))



    def forward(self, x, concat_with):
        if concat_with == None:
            if self.res:
                conv_x = self._net(x) + x
            else:
                conv_x = self._net(x)
        else:
            if self.res:
                conv_x = self._net(torch.cat([x, concat_with], dim=1)) + torch.cat([x, concat_with], dim=1)
            else:
                conv_x = self._net(torch.cat([x, concat_with], dim=1)) 

        return self.up_net(conv_x)

class SELayer_down(nn.Module):
    def __init__(self, H, W):
        super(SELayer_down, self).__init__()
        self.avg_pool_channel = nn.AdaptiveAvgPool1d(1)
        self.avg_pool_2d = nn.AdaptiveAvgPool2d((H//2, W//2))

    def forward(self, in_data, x):
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).view(b, -1, c)
        y = self.avg_pool_channel(x).view(b, h, w, 1)
        y = y.permute(0, 3, 1, 2)
        y = self.avg_pool_2d(y)
        return in_data * y.expand_as(in_data)





class DecoderBN(nn.Module):
    def __init__(self, num_features=128, lambda_val=1, res=True):
        super(DecoderBN, self).__init__()
        features = int(num_features)
        self.lambda_val = lambda_val

        self.se1_down = SELayer_down(120, 160)
        self.se2_down = SELayer_down(60, 80)
        self.se3_down = SELayer_down(30, 40)

        self.up1 = UpSampleBN(192, features, res)
        self.up2 = UpSampleBN(features + 96, features, res)
        self.up3 = UpSampleBN(features + 48, features, res)
        self.up4 = UpSampleBN(features + 24, features//2, res)


    def forward(self, features):
        x_block4, x_block3, x_block2, x_block1= features[3], features[2], features[1], features[0]

        x_block2_1 = self.lambda_val * self.se1_down(x_block2, x_block1) + (1-self.lambda_val) * x_block2
        x_block3_1 = self.lambda_val * self.se2_down(x_block3, x_block2_1) + (1-self.lambda_val) * x_block3
        x_block4_1 = self.lambda_val * self.se3_down(x_block4, x_block3_1) + (1-self.lambda_val) * x_block4

        x_d0 = self.up1(x_block4_1, None)
        x_d1 = self.up2(x_d0, x_block3_1)
        x_d2 = self.up3(x_d1, x_block2_1)
        x_d3 = self.up4(x_d2, x_block1)


        return x_d3


class Tode(nn.Module):
    def __init__(self, lambda_val = 1, res = True):
        super(Tode, self).__init__()

        self.encoder = SwinTransformer(patch_size=2, in_chans= 4, embed_dim=24)
        self.decoder = DecoderBN(num_features=128, lambda_val=lambda_val, res=res)


        self.final = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64, 64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, img, depth, **kwargs):
        n, h, w = depth.shape
        depth = depth.view(n, 1, h, w)

        # x = self._netin(torch.cat((img, depth), dim=1))
        encoder_x = self.encoder(torch.cat((img, depth), dim=1))
        decoder_x = self.decoder(encoder_x)

        out = self.final(decoder_x)


        return out


    @classmethod
    def build(cls, **kwargs):
 
        print('Building Encoder-Decoder model..', end='')
        m = cls(**kwargs)
        print('Done.')
        return m


if __name__ == '__main__':
    
    model = Tode.build(100)
    x = torch.rand(2, 3, 240, 320)
    y = torch.rand(2, 1, 240, 230)
    bins, pred = model(x,y)
    print(bins.shape, pred.shape)
