import torch.nn as nn
import torch.nn.functional as F
import torch
import copy, math

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.3):
        "Take in model size and number of heads."
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        nbatches, c, h, w = value.shape
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        # nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query.view(nbatches, 64*5*7), key.view(nbatches, 64*5*7), value.view(nbatches, 64*5*7)))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        y = self.linears[-1](x).view(nbatches, c, h, w)
        return y

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class BaseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None):
        super(BaseBlock, self).__init__()

        if padding is None:
            padding = (0, 0)
        else:
            padding = padding

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode='zeros')

        self.bn = nn.BatchNorm2d(in_channels)   # out_channels

        self.gelu = F.gelu

    def forward(self, x):
        # BN
        x = self.bn(x)
        # relu
        x = self.gelu(x)
        # conv
        x = self.conv(x)

        return x


class UP_BaseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None):
        super(UP_BaseBlock, self).__init__()

        if padding is None:
            padding = (0, 0)
        else:
            padding = padding

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)

        self.bn = nn.BatchNorm2d(in_channels)
        self.gelu = F.gelu

    def forward(self, x):
        # BN
        x = self.bn(x)
        # relu
        x = self.gelu(x)
        # conv
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, base1, base2, base3):
        super().__init__()
        self.base1 = base1
        self.base2 = base2
        self.base3 = base3
        self.relu = F.relu

    def forward(self, x):
        out1 = self.base1(x)
        y = self.base2(x)
        out2 = self.base3(y)
        out = out1 + out2

        return out


class BottleneckResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MultiHeadExternalAttention1(nn.Module):
    def __init__(self, num_head, d_model, c, h, w, dropout=0.3):
        "Take in model size and number of heads."
        super(MultiHeadExternalAttention1, self).__init__()
        assert d_model % num_head == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_head
        self.num_head = num_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.initial_query_input = nn.Parameter(torch.rand(1, w * h * c))

    def forward(self, key, value, mask=None):
        nbatches, c, h, w = value.shape
        query = self.initial_query_input.repeat(nbatches, 1)
        # key.shape[0]
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.num_head, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query.view(-1, c*h*w), key.view(-1, c*h*w), value.view(-1, c*h*w)))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.num_head * self.d_k)
        y = self.linears[-1](x).view(nbatches, c, h, w)
        return y

class psd2image(nn.Module):
    def __init__(self):
        super().__init__()
        self.res1 = ResBlock(BaseBlock(2, 8, (3, 3), (2, 2)), BaseBlock(2, 4, (3, 3), (2, 2)), BaseBlock(4, 8, (1, 1), (1, 1)))
        self.res2 = ResBlock(BaseBlock(8, 32, (3, 3), (2, 2)), BaseBlock(8, 16, (3, 3), (2, 2)), BaseBlock(16, 32, (1, 1), (1, 1)))
        self.res3 = ResBlock(BaseBlock(32, 128, (3, 3), (2, 2)), BaseBlock(32, 64, (3, 3), (2, 2)), BaseBlock(64, 128, (1, 1), (1, 1)))
        self.res4 = ResBlock(BaseBlock(128, 512, (3, 3), (2, 2)), BaseBlock(128, 256, (3, 3), (2, 2)), BaseBlock(256, 512, (1, 1), (1, 1)))
        self.res5 = ResBlock(BaseBlock(512, 1024, (3, 3), (2, 2)), BaseBlock(512, 1024, (3, 3), (2, 2)), BaseBlock(1024, 1024, (1, 1), (1, 1)))
        self.res6 = ResBlock(BaseBlock(1024, 2048, (3, 3), (2, 2)), BaseBlock(1024, 2048, (3, 3), (2, 2)), BaseBlock(2048, 2048, (1, 1), (1, 1)))
        self.dense2_1 = nn.Conv2d(2048, 64, 1, 1)
        self.attenlayer1 = MultiHeadExternalAttention1(16, 64*7*15, 64, 7, 15)   # head,  batchsize, head
        self.dense2_2 = nn.Conv2d(64, 2048, 1, 1)
        self.resup1 = ResBlock(UP_BaseBlock(2048, 1024, (5, 3), (3, 2)), UP_BaseBlock(2048, 1024, (5, 3), (3, 2)), UP_BaseBlock(1024, 1024, (1, 1), (1, 1)))
        self.resup2 = ResBlock(UP_BaseBlock(1024, 512, (3, 3), (2, 2)), UP_BaseBlock(1024, 512, (3, 3), (2, 2)), UP_BaseBlock(512, 512, (1, 1), (1, 1)))
        self.resup3 = ResBlock(UP_BaseBlock(512, 128, (3, 3), (2, 2)), UP_BaseBlock(512, 256, (3, 3), (2, 2)), UP_BaseBlock(256, 128, (1, 1), (1, 1)))
        self.resup4 = ResBlock(UP_BaseBlock(128, 32, (3, 3), (2, 2)), UP_BaseBlock(128, 64, (3, 3), (2, 2)), UP_BaseBlock(64, 32, (1, 1), (1, 1)))
        self.resup5 = ResBlock(UP_BaseBlock(32, 8, (3, 3), (2, 2)),
                               UP_BaseBlock(32, 16, (3, 3), (2, 2)),
                               UP_BaseBlock(16, 8, (1, 1), (1, 1)))
        self.resup6 = ResBlock(UP_BaseBlock(8, 1, (3, 3), (2, 2)),
                               UP_BaseBlock(8, 4, (3, 3), (2, 2)),
                               UP_BaseBlock(4, 1, (1, 1), (1, 1)))

    def forward(self, x):

        d1 = x.size(dim=0)
        x = x.view(d1, 2, 70, 1024, 8)
        x = x.permute(0, 1, 2, 4, 3).contiguous().view(d1, 2, 70 * 8, 8192 // 8)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.dense2_1(x)
        x = self.attenlayer1(x, x, None)
        x = self.dense2_2(x)
        x = self.resup1(x)
        x = self.resup2(x)
        x = self.resup3(x)
        x = self.resup4(x)
        x = self.resup5(x)
        x = self.resup6(x)
        x = F.interpolate(x, size=(1080, 980), mode='bilinear', align_corners=False)
        x = torch.clamp(x, min=0, max=1)

        return x


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model = psd2image()
    model.to(device)
    image = torch.randn(16, 2, 70, 8192)
    image = image.to(device)
    output = model(image)
    print(type(output))
    print(output.shape)