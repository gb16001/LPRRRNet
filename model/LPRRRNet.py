import torch
import torch.nn as nn
import torch.nn.functional as F



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

class GRU_2l_en_1l_de_res_(nn.Module):
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
            
        )
        self.linear1 = nn.Sequential(
            nn.Linear(36, 18),
            nn.BatchNorm1d(num_features=74),
            nn.ReLU(),
        )
        self.rnn_encoder=nn.GRU(74,74,bidirectional=True)
        self.rnn_decoder=nn.GRU(74,74,bidirectional=True)
        self.linear2=nn.Linear(74*2,74,)
        self.normL2=nn.BatchNorm1d(74)
        return
    def forward(self,x:torch.Tensor):
        logits_2s=self.backbone(x)
        logits_1s=self.linear1(logits_2s)
        logits_2t=logits_2s.permute(2,0,1).contiguous()
        logits_1t=logits_1s.permute(2,0,1).contiguous()
        _,hidden_0=self.rnn_encoder(logits_2t)
        
        y_hat,_=self.rnn_decoder(logits_1t,hidden_0)

        y_hat=y_hat.permute(1,0,2)
        y_hat=self.linear2(y_hat)
        y_hat=y_hat.permute(0,2,1)
        y_hat=self.normL2(y_hat)
        y_hat=F.relu(y_hat)
        return y_hat+logits_1s #softmax to property

class shuff_mob_gru(nn.Module):
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
            nn.Conv2d(81,class_num,(1,3),1),
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),
            nn.Flatten(2,3),
            
        )
        self.linear1 = nn.Sequential(
            nn.Linear(36, 18),
            nn.BatchNorm1d(num_features=class_num),
            nn.ReLU(),
        )
        self.rnn_encoder=nn.GRU(class_num,class_num,bidirectional=True)
        self.rnn_decoder=nn.GRU(class_num,class_num,bidirectional=True)
        self.linear2=nn.Linear(class_num*2,class_num,)
        self.normL2=nn.BatchNorm1d(class_num)
        return
    def forward(self,x:torch.Tensor):
        logits_2s=self.backbone(x)
        logits_1s=self.linear1(logits_2s)
        logits_2t=logits_2s.permute(2,0,1).contiguous()
        logits_1t=logits_1s.permute(2,0,1).contiguous()
        _,hidden_0=self.rnn_encoder(logits_2t)
        
        y_hat,_=self.rnn_decoder(logits_1t,hidden_0)

        y_hat=y_hat.permute(1,0,2)
        y_hat=self.linear2(y_hat)
        y_hat=y_hat.permute(0,2,1)
        y_hat=self.normL2(y_hat)
        y_hat=F.relu(y_hat)
        return y_hat+logits_1s #softmax to property

class LPRRRNet(nn.Module):
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
            nn.Conv2d(81,class_num,(1,3),1),
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),
            nn.Flatten(2,3),
            
        )
        self.linear1 = nn.Sequential(
            nn.Linear(36, 18),
            nn.BatchNorm1d(num_features=class_num),
            nn.ReLU(),
        )
        self.rnn_encoder=nn.GRU(class_num,class_num,bidirectional=True)
        self.rnn_decoder=nn.GRU(class_num,class_num,bidirectional=True)
        self.linear2=nn.Linear(class_num*2,class_num,)
        self.normL2=nn.BatchNorm1d(class_num)
        return
    def forward(self,x:torch.Tensor):
        logits_2s=self.backbone(x)
        logits_1s=self.linear1(logits_2s)
        logits_2t=logits_2s.permute(2,0,1).contiguous()
        logits_1t=logits_1s.permute(2,0,1).contiguous()
        _,hidden_0=self.rnn_encoder(logits_2t)
        
        y_hat,_=self.rnn_decoder(logits_1t,hidden_0)

        y_hat=y_hat.permute(1,0,2)
        y_hat=self.linear2(y_hat)
        y_hat=y_hat.permute(0,2,1)
        y_hat=self.normL2(y_hat)
        y_hat=F.relu(y_hat)
        return y_hat+logits_1s #softmax to property

def __test():
    net=LPRRRNet(class_num=74)
    x=torch.randn(64,3,24,94)
    y_hat=net(x)
    print(y_hat.size())
    return


class Shuffle_Mobile_Block(nn.Module): # img transfer
    def __init__(self, channels, groups) -> None:
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(channels,channels,3,1,1,groups=groups),
            ChannelShuffle(groups),
            nn.Conv2d(channels,channels,3,1,1),
            nn.Conv2d(channels,channels,1,1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        return

    def forward(self, x: torch.Tensor):
        out = x + self.block(x)
        return out
class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        # Reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        # Transpose
        x = x.transpose(1, 2).contiguous()
        # Flatten
        x = x.view(batchsize, -1, height, width)
        return x
def __test_block():
    block=Shuffle_Mobile_Block(27,9)
    x=torch.randn(2,27,24,94)
    y=block(x)
    print(y.size())
    return

def __compair_shuffle():
    x= torch.randn(2,8,10,20)
    torch_m=nn.ChannelShuffle(4)
    my_m=ChannelShuffle(4)
    compare=torch.equal(torch_m(x),my_m(x))
    print(f"nn.ChannelShuffle==ChannelShuffle: {compare}")
    return compare





def norm_init_weights(m):# normalize init weight
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)

def init_net_weight(lprnet:nn.Module,args):
    if args.pretrained_model:
        # load pretrained model
        lprnet.load_state_dict(torch.load(args.pretrained_model))
        print("load pretrained model successful!")
    elif False:
        #TODO backbone load_state_dict,container weights_init
        None
    else:
        for idx,m in enumerate(lprnet.modules()):
            m.apply(norm_init_weights)        
        print("initial net weights successful!")
    return 

if __name__=="__main__":
    # net= LPRNet(lpr_max_len=18, phase=False, class_num=74, dropout_rate=0.5)
    __test()
    # test_block()
    pass
