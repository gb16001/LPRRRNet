import torch
import torch.nn as nn
import torch.nn.functional as F


def rnn():
    # rnn=nn.LSTM(input_size=74,hidden_size=512)
    rnn=nn.GRU(input_size=74,hidden_size=512,bidirectional=False,device=None)
    rnn.eval()
    h_0=torch.randn([1,512])# bio*layers,N,ch
    in_vect=torch.randn([10,74])# L,N,ch
    out_vect,h_n=rnn(in_vect)
    print(out_vect.size(),h_n.size(),torch.equal(out_vect[-1],h_n[0]))
    return

def trans():
    transformer_model=nn.Transformer(norm_first=True)
    src = torch.rand((10, 32, 512))
    tgt = torch.rand((20, 32, 512))
    out = transformer_model(src, tgt)
    print(out.size())
    return

class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
    def forward(self, x):
        return self.block(x)
class shuffle_mobile_block(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        return
    def forward(self,x):
        return
    
class LPRNet(nn.Module):
    def __init__(self, lpr_max_len, phase, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        # self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1), # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            small_basic_block(ch_in=64, ch_out=128),    # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            small_basic_block(ch_in=64, ch_out=256),   # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),   # *** 11 ***
            nn.BatchNorm2d(num_features=256),   # 12
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1), # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # *** 22 ***
        )
        self.neck=None # TODO: extruct the neck net
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=448+self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
            # nn.BatchNorm2d(num_features=self.class_num),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=self.class_num, out_channels=self.lpr_max_len+1, kernel_size=3, stride=2),
            # nn.ReLU(),
        )

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]: # [2, 4, 8, 11, 22]
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.head(x)
        logits = torch.mean(x, dim=2)

        return logits
class GRU_ende_res(nn.Module):
    def __init__(self, class_num:int) -> None:
        super().__init__()
        self.classNum=class_num
        self.backbone=nn.Sequential(
            nn.Conv2d(3,9,3,1),
            nn.BatchNorm2d(num_features=9),
            nn.ReLU(),
            nn.Conv2d(9,27,3,2),
            nn.BatchNorm2d(num_features=27),
            nn.ReLU(),
            nn.Conv2d(27,81,3,2),
            nn.BatchNorm2d(num_features=81),
            nn.ReLU(),
            # nn.MaxPool2d((3,1),stride=1),
            nn.Conv2d(81,81,(3,3),1,groups=9),
            nn.BatchNorm2d(num_features=81),
            nn.Conv2d(81,81,1,1),
            nn.BatchNorm2d(num_features=81),
            nn.ReLU(),
            nn.Conv2d(81,74,(1,3),1),
            nn.BatchNorm2d(num_features=74),
            nn.ReLU(),
            nn.Flatten(2,3),
            nn.Linear(36,18),
            nn.BatchNorm1d(num_features=74),
            nn.ReLU(),
        )
        self.rnn_encoder=nn.GRU(74,74,bidirectional=True)
        self.rnn_decoder=nn.GRU(74,74,bidirectional=True)
        self.linear2=nn.Linear(74*2,74,)
        self.normL2=nn.BatchNorm1d(74)
        return
    def forward(self,x:torch.Tensor):
        logits_s=self.backbone(x)
        
        logits_t=logits_s.permute(2,0,1).contiguous()
        _,hidden_0=self.rnn_encoder(logits_t)
        y_hat,_=self.rnn_decoder(logits_t,hidden_0)

        y_hat=y_hat.permute(1,0,2)
        y_hat=self.linear2(y_hat)
        y_hat=y_hat.permute(0,2,1)
        y_hat=self.normL2(y_hat)
        y_hat=F.relu(y_hat)
        return y_hat+logits_s #softmax to property

class myNet(nn.Module):# TODO rename class
    def __init__(self, class_num:int) -> None:
        super().__init__()
        self.classNum=class_num
        self.backbone=nn.Sequential(
            nn.Conv2d(3,9,3,1),
            nn.BatchNorm2d(num_features=9),
            nn.ReLU(),
            nn.Conv2d(9,27,3,2),
            nn.BatchNorm2d(num_features=27),
            nn.ReLU(),
            nn.Conv2d(27,81,3,2),
            nn.BatchNorm2d(num_features=81),
            nn.ReLU(),
            # nn.MaxPool2d((3,1),stride=1),
            nn.Conv2d(81,81,(3,3),1,groups=9),
            nn.BatchNorm2d(num_features=81),
            nn.Conv2d(81,81,1,1),
            nn.BatchNorm2d(num_features=81),
            nn.ReLU(),
            nn.Conv2d(81,74,(1,3),1),
            nn.BatchNorm2d(num_features=74),
            nn.ReLU(),
            nn.Flatten(2,3),
            nn.Linear(36,18),
            nn.BatchNorm1d(num_features=74),
            nn.ReLU(),
        )
        self.rnn_encoder=nn.GRU(74,74,bidirectional=True)
        self.rnn_decoder=nn.GRU(74,74,bidirectional=True)
        self.linear2=nn.Linear(74*2,74,)
        self.normL2=nn.BatchNorm1d(74)
        return
    def forward(self,x:torch.Tensor):
        logits_s=self.backbone(x)
        
        logits_t=logits_s.permute(2,0,1).contiguous()
        _,hidden_0=self.rnn_encoder(logits_t)
        y_hat,_=self.rnn_decoder(logits_t,hidden_0)

        y_hat=y_hat.permute(1,0,2)
        y_hat=self.linear2(y_hat)
        y_hat=y_hat.permute(0,2,1)
        y_hat=self.normL2(y_hat)
        y_hat=F.relu(y_hat)
        return y_hat+logits_s #softmax to property

def test():
    net=myNet(class_num=74)
    x=torch.randn(64,3,24,94)
    y_hat=net(x)
    print(y_hat.size())
    return



# ShuffleNet Block
class ShuffleNetBlock(nn.Module):
    def __init__(self, inp, oup, stride):
        super(ShuffleNetBlock, self).__init__()
        self.stride = stride
        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features * 2)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if self.stride > 1 else branch_features, branch_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_features, branch_features, 3, stride, 1, groups=branch_features, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x:torch.Tensor):
        if self.stride > 1:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
        else:
            x1, x2 = x.chunk(2, dim=1)
            x2 = self.branch2(x2)
        out = torch.cat((x1, x2), 1)
        out = F.channel_shuffle(out, 2)
        return out


# MobileNet Block
class MobileNetBlock(nn.Module):
    def __init__(self, inp, oup, stride):
        super(MobileNetBlock, self).__init__()
        self.stride = stride
        self.block = nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
        return

    def forward(self, x):
        
        return self.block(x)

def test_shuffle_mobile():
    # Example Usage:
    # Define input tensor
    inp_tensor = torch.randn(1, 24, 56, 56)

    # Create ShuffleNet block
    shuffle_block = ShuffleNetBlock(inp=24, oup=48, stride=1)
    shuffle_output = shuffle_block(inp_tensor)

    # Create MobileNet block
    mobile_block = MobileNetBlock(inp=24, oup=48, stride=2)
    mobile_output = mobile_block(inp_tensor)

    print("ShuffleNet Block Output Shape:", shuffle_output.shape)
    print("MobileNet Block Output Shape:", mobile_output.shape)
    return


if __name__=="__main__":
    # net= LPRNet(lpr_max_len=18, phase=False, class_num=74, dropout_rate=0.5)
    test()
    pass
