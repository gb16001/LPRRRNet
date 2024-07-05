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

class Shuffle_Mobile_Block(nn.Module): # img transfer
    def __init__(self, channels, groups) -> None:
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(channels,channels,3,1,1,groups=groups),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            ChannelShuffle(groups),
            # nn.Conv2d(channels,channels,3,1,1),
            nn.Conv2d(channels,channels,3,1,1,groups=groups),
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


class Backbone:
    def LPRnet(dropout_rate=0.5,char_classNum=74):
        return nn.Sequential(
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
            nn.Conv2d(in_channels=256, out_channels=char_classNum, kernel_size=(13, 1), stride=1), # 20
            nn.BatchNorm2d(num_features=char_classNum),
            nn.ReLU(),  # *** 22 ***
        )
    def grouped(char_classNum=74):
        return nn.Sequential(
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
            nn.Conv2d(81,char_classNum,(1,3),1),
            nn.BatchNorm2d(num_features=char_classNum),
            nn.ReLU(),
        )
    def O_fix(char_classNum=74):
        return nn.Sequential(
            nn.Conv2d(3,9,3,1),
            nn.BatchNorm2d(num_features=9),
            nn.ReLU(),
            Shuffle_Mobile_Block(9,3),
            nn.Conv2d(9,27,3,1),
            nn.BatchNorm2d(num_features=27),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1),
            Shuffle_Mobile_Block(27,9),
            nn.Conv2d(27,81,3,1),
            nn.BatchNorm2d(num_features=81),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1),
            # nn.MaxPool2d((3,1),stride=1),
            nn.Conv2d(81,81,(3,3),1,groups=9),
            nn.BatchNorm2d(num_features=81),
            nn.Conv2d(81,81,1,1),
            nn.BatchNorm2d(num_features=81),
            nn.ReLU(),
            Shuffle_Mobile_Block(81,9),
            nn.Conv2d(81,char_classNum,(1,5),1,(0,1)),# expand kernel feild, to fix O recognition
            nn.BatchNorm2d(num_features=char_classNum),
            nn.ReLU(),
        )
    def M_S(char_classNum=74):
        return nn.Sequential(
            nn.Conv2d(3,9,3,1),
            nn.BatchNorm2d(num_features=9),
            nn.ReLU(),
            Shuffle_Mobile_Block(9,3),
            nn.Conv2d(9,27,3,1),
            nn.BatchNorm2d(num_features=27),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1),
            Shuffle_Mobile_Block(27,9),
            nn.Conv2d(27,81,3,1),
            nn.BatchNorm2d(num_features=81),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1),
            # nn.MaxPool2d((3,1),stride=1),
            nn.Conv2d(81,81,(3,3),1,groups=9),
            nn.BatchNorm2d(num_features=81),
            nn.Conv2d(81,81,1,1),
            nn.BatchNorm2d(num_features=81),
            nn.ReLU(),
            Shuffle_Mobile_Block(81,9),
            nn.Conv2d(81,char_classNum,(1,3),1),
            nn.BatchNorm2d(num_features=char_classNum),
            nn.ReLU(),
        )
    def lprImg2FM81():
        return nn.Sequential(
            nn.Conv2d(3,9,3,1),
            nn.BatchNorm2d(num_features=9),
            nn.ReLU(),
            Shuffle_Mobile_Block(9,3),
            nn.Conv2d(9,27,3,1),
            nn.BatchNorm2d(num_features=27),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1),
            Shuffle_Mobile_Block(27,9),
            nn.Conv2d(27,81,3,1),
            nn.BatchNorm2d(num_features=81),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1),
            # nn.MaxPool2d((3,1),stride=1),
            nn.Conv2d(81,81,(3,3),1,groups=9),
            nn.BatchNorm2d(num_features=81),
            nn.Conv2d(81,81,1,1),
            nn.BatchNorm2d(num_features=81),
            nn.ReLU(),
            Shuffle_Mobile_Block(81,9),
        )
    def FM81_2_charactor(char_classNum:int=74):
        return nn.Sequential(
            nn.Conv2d(81, char_classNum, (1, 5), 1, (0, 1)),  # expand kernel feild, to fix O recognition
            nn.BatchNorm2d(num_features=char_classNum),
            nn.ReLU(),
        )
    pass
class Neck:
    def flate():
        return nn.Flatten(2,3)
    def lprnet():
        # use class LPRNet derectly
        return
    pass
class Head:
    def lprnet():
        # use class LPRNet derectly
        return

    def spatialDense(char_classNum:int=74):
        return nn.Sequential(
            nn.Linear(36, 18),
            nn.BatchNorm1d(num_features=char_classNum),
            nn.ReLU(),
        )

    class resRNN(nn.Module):
        def __init__(self, char_classNum:int=74) -> None:
            super().__init__()
            self.spatialDense = nn.Sequential(
            nn.Linear(36, 18),
            nn.BatchNorm1d(num_features=char_classNum),
            nn.ReLU(),
            )
            self.rnn_encoder=nn.GRU(char_classNum,char_classNum,bidirectional=True)
            self.rnn_decoder=nn.GRU(char_classNum,char_classNum,bidirectional=True)
            self.channelDense=nn.Linear(char_classNum*2,char_classNum,)
            self.normL2=nn.BatchNorm1d(char_classNum)
            return
        def forward(self,logits_2s:torch.Tensor):# inshape(bz,class,36)
            logits_1s=self.spatialDense(logits_2s)
            logits_2t=logits_2s.permute(2,0,1).contiguous()# (N,B,C)
            logits_1t=logits_1s.permute(2,0,1).contiguous()
            _,hidden_0=self.rnn_encoder(logits_2t)

            y_hat,_=self.rnn_decoder(logits_1t,hidden_0)

            y_hat=y_hat.permute(1,0,2)
            y_hat=self.channelDense(y_hat)
            y_hat=y_hat.permute(0,2,1)
            y_hat=self.normL2(y_hat)
            y_hat=F.relu(y_hat)
            return y_hat+logits_1s #softmax to property

    class attnRNN(nn.Module):
        def __init__(self, char_classNum: int) -> None:
            super().__init__()
            self.classNum_char=char_classNum
            self.rnn_encoder = nn.GRU(char_classNum, char_classNum, bidirectional=True)
            self.rnn_decoder = nn.GRU(char_classNum, char_classNum,2)
            self.channelDense = nn.Sequential(
                nn.Linear(char_classNum , char_classNum),
                nn.LayerNorm(char_classNum),
                nn.ReLU(),
            )
            return

        def get_H0(self, logits_2s: torch.Tensor):
            r"""logits_2s:(bz,class,36)"""
            logits_2t = logits_2s.permute(2, 0, 1).contiguous()  # (N,B,C)
            _, hidden_0 = self.rnn_encoder(logits_2t)
            return hidden_0

        def forward(self, Q_t:torch.Tensor, hidden_0):
            r"""Q_t:int(N,B)"""
            x=F.one_hot(Q_t.long(),self.classNum_char).to(torch.float32)
            y_hat, hidden = self.rnn_decoder(x, hidden_0)
            y_hat = self.channelDense(y_hat)
            return y_hat, hidden

    def classifier(LPR_classNum:int=7):
        '''x:(bz,81,2,20)'''
        return nn.Sequential(
                nn.Conv2d(81,27,2,2,groups=9),
                nn.BatchNorm2d(27),
                nn.ReLU(),
                nn.Conv2d(27,9,(1,2),(1,2),groups=9),
                nn.BatchNorm2d(9),
                nn.ReLU(),
                nn.Conv2d(9,LPR_classNum,1),
                nn.ReLU(),
                nn.AvgPool2d((1,5)),
                nn.Flatten()
            )

    pass


class LPRNet(nn.Module):
    def __init__(self, lpr_max_len, phase, char_classNum, dropout_rate):
        super(LPRNet, self).__init__()
        # self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.char_classNum = char_classNum
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
            nn.Conv2d(in_channels=256, out_channels=char_classNum, kernel_size=(13, 1), stride=1), # 20
            nn.BatchNorm2d(num_features=char_classNum),
            nn.ReLU(),  # *** 22 ***
        )
        self.neck=None # TODO: extruct the neck net
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=448+self.char_classNum, out_channels=self.char_classNum, kernel_size=(1, 1), stride=(1, 1)),
            # nn.BatchNorm2d(num_features=self.char_classNum),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=self.char_classNum, out_channels=self.lpr_max_len+1, kernel_size=3, stride=2),
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
    def __init__(self, char_classNum:int) -> None:
        super().__init__()
        self.classNum=char_classNum
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
    def __init__(self, char_classNum:int) -> None:
        super().__init__()
        self.classNum=char_classNum
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
    def __init__(self, char_classNum:int) -> None:
        super().__init__()
        self.classNum=char_classNum
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
            nn.Conv2d(81,char_classNum,(1,3),1),
            nn.BatchNorm2d(num_features=char_classNum),
            nn.ReLU(),
            nn.Flatten(2,3),
            
        )
        self.linear1 = nn.Sequential(
            nn.Linear(36, 18),
            nn.BatchNorm1d(num_features=char_classNum),
            nn.ReLU(),
        )
        self.rnn_encoder=nn.GRU(char_classNum,char_classNum,bidirectional=True)
        self.rnn_decoder=nn.GRU(char_classNum,char_classNum,bidirectional=True)
        self.linear2=nn.Linear(char_classNum*2,char_classNum,)
        self.normL2=nn.BatchNorm1d(char_classNum)
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
    def __init__(self, char_classNum:int) -> None:
        super().__init__()
        self.classNum=char_classNum
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
            nn.Conv2d(81,char_classNum,(1,3),1),
            nn.BatchNorm2d(num_features=char_classNum),
            nn.ReLU(),
            nn.Flatten(2,3),
            
        )
        self.linear1 = nn.Sequential(
            nn.Linear(36, 18),
            nn.BatchNorm1d(num_features=char_classNum),
            nn.ReLU(),
        )
        self.rnn_encoder=nn.GRU(char_classNum,char_classNum,bidirectional=True)
        self.rnn_decoder=nn.GRU(char_classNum,char_classNum,bidirectional=True)
        self.linear2=nn.Linear(char_classNum*2,char_classNum,)
        self.normL2=nn.BatchNorm1d(char_classNum)
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
class O_fix(nn.Module):
    def __init__(self,class_num:int) -> None:
        super().__init__()
        back=Backbone.O_fix(class_num)
        neck=Neck.flate()
        head=Head.resRNN(class_num)
        self.net=nn.Sequential(back,neck,head)
        return
    def forward(self,x:torch.Tensor):
        return self.net(x)

class Classify(nn.Module):
    def __init__(self,char_classNum:int) -> None:
        super().__init__()
        self.back0=Backbone.lprImg2FM81()
        self.back1=Backbone.FM81_2_charactor(char_classNum)
        self.neck=Neck.flate()
        self.head=Head.resRNN(char_classNum)
        self.head_class=Head.classifier()
        # self.net=nn.Sequential(back0,head_class)
        return
    def forward(self,x:torch.Tensor):
        x=self.back0(x)
        lprClass=self.head_class(x)
        x=self.back1(x)
        x=self.neck(x)
        x=self.head(x)
        return x,lprClass

class attn_predictNet(nn.Module):
    def __init__(self,class_num:int) -> None:
        super().__init__()
        self.back=Backbone.O_fix(class_num)
        self.neck=Neck.flate()
        self.head=Head.attnRNN(class_num)
        # self.net=nn.Sequential(back,neck,head)
        return
    def forward(self,x:torch.Tensor):
        """return H_0 for RNN decoder"""
        x=self.back(x)
        x=self.neck(x)
        hidden_0=self.head.get_H0(x)
        return hidden_0
    def forward_dec(self,x:torch.Tensor,h_0):
        '''x:int(N,B)'''
        y_hat,hidden=self.head(x,h_0)
        return y_hat.permute(0,2,1),hidden #(N,C,BZ)
    def forward_train(self,img:torch.Tensor,x:torch.Tensor):
        r'''img:(bz,ch,y,x) x:int(N,B)'''
        H_0=self(img)
        y_hat,_=self.forward_dec(x,H_0)
        return y_hat# (N,C,BZ)
    pass
class myNet(nn.Module):
    def __init__(self,class_num:int) -> None:
        super().__init__()
        self.back=Backbone.O_fix(class_num)
        self.neck=Neck.flate()
        self.head=Head.attnRNN(class_num)
        # self.net=nn.Sequential(back,neck,head)
        return
    def forward(self,x:torch.Tensor):
        """return H_0 for RNN decoder"""
        x=self.back(x)
        x=self.neck(x)
        hidden_0=self.head.get_H0(x)
        return hidden_0
    def forward_dec(self,x:torch.Tensor,h_0):
        '''x:int(N,B)'''
        y_hat,hidden=self.head(x,h_0)
        return y_hat.permute(0,2,1),hidden #(N,C,BZ)
    def forward_train(self,img:torch.Tensor,x:torch.Tensor):
        r'''img:(bz,ch,y,x) x:int(N,B)'''
        H_0=self(img)
        y_hat,_=self.forward_dec(x,H_0)
        return y_hat# (N,C,BZ)
    pass

def __test():
    classNum_char=74
    batchSize=4
    maxLength=8
    net=myNet(classNum_char)
    # net=LPRRRNet(char_classNum=74)
    img=torch.randn(batchSize,3,24,94)
    def gen_seq():
        # 生成随机序列
        X = torch.randint(0, classNum_char, (maxLength, batchSize))
        Y = torch.randint(0, classNum_char, (maxLength, batchSize))
        # 确保X的第一个元素是73
        X[0, :] = 73
        # 确保Y的最后一个元素是73
        Y[-1, :] = 73
        # 确保X的后面的元素与Y的前面的元素相同
        X[1:, :] = Y[:-1, :]
        return X,Y
    X,Y=gen_seq()
        
    # for i in range(batchSize):
    #     X[1:, i] = Y[:-1, i]

    print("X:", X)
    print("Y:", Y)
    # forward
    H_0=net(img)
    y_hat:torch.Tensor=net.head(X,H_0)
    y_hat=y_hat.permute(0,2,1)
    criterion = nn.CrossEntropyLoss()
    loss=criterion(y_hat, Y)
    loss.backward()
    print(loss)
    # try:
    #     y_hat,class_hat=net(img)
    # except ValueError:
    #     y_hat,class_hat=net(img),None
    # print(y_hat.size(),class_hat.size() if class_hat is not None else None)
    # if class_hat is not None:
    #     labels=torch.randint(0, 7, (8,))
    #     # 定义交叉熵损失函数
    #     criterion = nn.CrossEntropyLoss()
    #     # 计算损失
    #     loss = criterion(class_hat, labels)
    #     print("Loss:", loss.item())
    #     loss.backward()
    return
if __name__=="__main__":
    # net= LPRNet(lpr_max_len=18, phase=False, char_classNum=74, dropout_rate=0.5)
    __test()
    # test_block()
    pass
