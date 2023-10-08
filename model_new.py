import torch
import torch.nn as nn
from block_new import get_sinusoid_encoding_table, get_attn_key_pad_mask, get_non_pad_mask, \
    get_subsequent_mask, EncoderLayer
from config import PAD, KS, Fea_PLUS
import torch.nn.functional as F
from loss import cca_loss
from config import *


class EEGEncoder(nn.Module):
    def __init__(
            self,
            d_feature,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout):
        super().__init__()

        n_position = d_feature + 1
        self.src_word_emb = nn.Conv1d(1, d_model, kernel_size=KS, padding=int((KS - 1) / 2))

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        self.eeg_layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout, d_feature)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos):

        non_pad_mask = get_non_pad_mask(src_seq)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        enc_output = src_seq.unsqueeze(1)
        enc_output = self.src_word_emb(enc_output)
        enc_output = enc_output.transpose(1, 2)
        enc_output.add_(self.position_enc(src_pos))

        for enc_layer in self.eeg_layer_stack:
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output,
    
class TextEncoder(nn.Module):
    def __init__(
            self,
            d_feature,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout):
        super().__init__()

        n_position = d_feature + 1
        self.src_word_emb = nn.Conv1d(1, d_model, kernel_size=KS, padding=int((KS - 1) / 2))

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        self.text_layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout, d_feature)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos):

        non_pad_mask = get_non_pad_mask(src_seq)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        enc_output = src_seq.unsqueeze(1)
        enc_output = self.src_word_emb(enc_output)
        enc_output = enc_output.transpose(1, 2)
        enc_output.add_(self.position_enc(src_pos))

        for enc_layer in self.text_layer_stack:
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output,
    
class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''
    def __init__(
            self, device, d_feature_text, d_feature_eeg, d_model, 
            d_inner, n_layers, n_head, d_k, d_v, dropout, class_num, args):

        super().__init__()

        if args.modality == 'text':
            self.text_encoder = TextEncoder(d_feature_text, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)
            self.linear1_cov_text = nn.Conv1d(d_feature_text, 1, kernel_size=1)

        elif args.modality == 'eeg':
            self.eeg_encoder = EEGEncoder(d_feature_eeg, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)
            self.linear1_cov_eeg = nn.Conv1d(d_feature_eeg, 1, kernel_size=1)
        else:
            self.text_encoder = TextEncoder(d_feature_text, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)
            self.eeg_encoder = EEGEncoder(d_feature_eeg, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)
            self.linear1_cov_fusion = nn.Conv1d(d_feature_text + d_feature_eeg, 1, kernel_size = 1)
            self.text_projection = nn.Conv1d(d_feature_text, 1, kernel_size = 1)
            self.eeg_projection = nn.Conv1d(d_feature_eeg, 1, kernel_size = 1)
            
            
        self.device = device
        self.args = args
        self.linear1_linear = nn.Linear(d_model, class_num)

    def forward(self, text_src_seq = None, eeg_src_seq = None):
        
        if (text_src_seq != None) and (eeg_src_seq == None):
            b_text, l_text = text_src_seq.size()
            src_pos_text = torch.LongTensor(
                [list(range(1, l_text + 1)) for i in range(b_text)]).to(self.device)
            
            enc_output_text, *_ = self.text_encoder(text_src_seq, src_pos_text)
            
            res_text = self.linear1_cov_text(enc_output_text)
            res_text = res_text.contiguous().view(res_text.size()[0], -1)
            res_text = self.linear1_linear(res_text)
            return res_text
        
        elif (eeg_src_seq != None) and (text_src_seq == None):
            b_eeg, l_eeg = eeg_src_seq.size()
            src_pos_eeg = torch.LongTensor(
                [list(range(1, l_eeg + 1)) for i in range(b_eeg)]).to(self.device)
            
            enc_output_eeg, *_ = self.eeg_encoder(eeg_src_seq, src_pos_eeg)
            
            res_eeg = self.linear1_cov_eeg(enc_output_eeg)
            res_eeg = res_eeg.contiguous().view(res_eeg.size()[0], -1)
            res_eeg = self.linear1_linear(res_eeg)     
            return res_eeg   
        
        elif (text_src_seq != None) and (eeg_src_seq != None):
            b_text, l_text = text_src_seq.size()
            b_eeg, l_eeg = eeg_src_seq.size()
            
            src_pos_text = torch.LongTensor(
            [list(range(1, l_text + 1)) for i in range(b_text)]).to(self.device)
            src_pos_eeg = torch.LongTensor(
            [list(range(1, l_eeg + 1)) for i in range(b_eeg)]).to(self.device)
        

            enc_output_text, *_ = self.text_encoder(text_src_seq, src_pos_text)
            enc_output_eeg, *_ = self.eeg_encoder(eeg_src_seq, src_pos_eeg)   
            projected_text = self.text_projection(enc_output_text)
            projected_eeg = self.eeg_projection(enc_output_eeg)
            
            concat_enc = torch.cat((enc_output_text, enc_output_eeg), dim = 1)
            
            res = self.linear1_cov_fusion(concat_enc)
            res = res.contiguous().view(res.size()[0], -1)
            res = self.linear1_linear(res)
            
            # FOR CCA
            projected_text = torch.squeeze(projected_text, dim=1)
            projected_eeg = torch.squeeze(projected_eeg, dim=1)
            return res, projected_eeg, projected_text


class MLP(nn.Module):
    def __init__(self, d_feature_text, d_feature_eeg, layer2, layer3, layer4, class_num, dropout, args):
        super(MLP, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.args = args
        if args.modality == 'text':
            self.l1_text = nn.Linear(d_feature_text, layer2, bias = False)
            self.l2_text = nn.Linear(layer2, layer3, bias = False)
            self.l3_text = nn.Linear(layer3, layer4, bias = False)
            self.l4_text = nn.Linear(layer4, class_num, bias = False)
            
        elif args.modality == 'eeg':
            self.l1_eeg = nn.Linear(d_feature_eeg, layer2, bias = False)
            self.l2_eeg = nn.Linear(layer2, layer3, bias = False)
            self.l3_eeg = nn.Linear(layer3, layer4, bias = False)
            self.l4_eeg = nn.Linear(layer4, class_num, bias = False)
            
        elif args.modality == 'fusion':
            self.l1_text = nn.Linear(d_feature_text, layer2, bias = False)
            self.l2_text = nn.Linear(layer2, layer3, bias = False)
            self.l3_text = nn.Linear(layer3, layer4, bias = False)
            
            self.l1_eeg = nn.Linear(d_feature_eeg, layer2, bias = False)
            self.l2_eeg = nn.Linear(layer2, layer3, bias = False)
            self.l3_eeg = nn.Linear(layer3, layer4, bias = False)
            
            self.l4_fusion = nn.Linear(layer4+layer4, class_num, bias = False)
            
        
        
                
    def forward(self, text_src_seq = None, eeg_src_seq = None):
        
        if (text_src_seq != None) and (eeg_src_seq == None):
            out_text = self.relu(self.l1_text(text_src_seq))
            out_text = self.dropout(out_text)
            out_text = self.relu(self.l2_text(out_text))
            out_text = self.dropout(out_text)
            out_text = self.relu(self.l3_text(out_text))
            out_text = self.dropout(out_text)
            out_text = self.l4_text(out_text)
            
            return out_text
        
        elif (text_src_seq == None) and (eeg_src_seq != None):
            out_eeg = self.relu(self.l1_eeg(eeg_src_seq))
            out_eeg = self.dropout(out_eeg)
            out_eeg = self.relu(self.l2_eeg(out_eeg))
            out_eeg = self.dropout(out_eeg)
            out_eeg = self.relu(self.l3_eeg(out_eeg))
            out_eeg = self.dropout(out_eeg)
            out_eeg = self.l4_eeg(out_eeg)
            
            return out_eeg
        
        elif (text_src_seq != None) and (eeg_src_seq != None):
            out_eeg = self.relu(self.l1_eeg(eeg_src_seq))
            out_eeg = self.dropout(out_eeg)
            out_eeg = self.relu(self.l2_eeg(out_eeg))
            out_eeg = self.dropout(out_eeg)
            out_eeg = self.relu(self.l3_eeg(out_eeg))
            out_eeg = self.dropout(out_eeg)
            
            out_text = self.relu(self.l1_text(text_src_seq))
            out_text = self.dropout(out_text)
            out_text = self.relu(self.l2_text(out_text))
            out_text = self.dropout(out_text)
            out_text = self.relu(self.l3_text(out_text))
            out_text = self.dropout(out_text)
            
            concat_feat = torch.cat((out_text, out_eeg), dim = 1)
            
            out_fusion = self.l4_fusion(concat_feat)
            
            return out_fusion
    
    
    
    
# The below code is from https://github.com/hsd1503/resnet1d/blob/master/resnet1d.py
# Thank you for the code! 
class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)

        return net
        
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        
        return net
    
class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)
                
        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        if self.downsample:
            identity = self.max_pool(identity)
            
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        out += identity

        return out
    
class ResNet1D(nn.Module):

    def __init__(self, in_channels, d_feature_eeg, d_feature_text, kernel_size, stride, groups, n_block, n_classes, args,\
        downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, verbose=False):
        super(ResNet1D, self).__init__()
        
        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do
        self.d_feature_eeg = d_feature_eeg
        self.d_feature_text = d_feature_text
        self.args = args

        self.downsample_gap = downsample_gap 
        self.increasefilter_gap = increasefilter_gap 
        if args.modality == 'text':
            self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=d_feature_text, kernel_size=self.kernel_size, stride=1)
            self.first_block_bn = nn.BatchNorm1d(d_feature_text)
            out_channels = d_feature_text
        elif args.modality == 'eeg':
            self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=d_feature_eeg, kernel_size=self.kernel_size, stride=1)
            self.first_block_bn = nn.BatchNorm1d(d_feature_eeg)
            out_channels = d_feature_eeg
        
        self.first_block_relu = nn.ReLU()
        
                
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
       
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            if is_first_block:
                if args.modality == 'eeg':
                    in_channels = d_feature_eeg
                elif args.modality == 'text':
                    in_channels = d_feature_text
                out_channels = in_channels
            else:
                if args.modality == 'eeg':
                    in_channels = int(d_feature_eeg*2**((i_block-1)//self.increasefilter_gap))
                    
                elif args.modality == 'text':
                    in_channels = int(d_feature_text*2**((i_block-1)//self.increasefilter_gap))
                    
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)

        self.dense = nn.Linear(out_channels, n_classes)

        
    def forward(self, text_src_seq = None, eeg_src_seq = None):
        if self.args.modality == 'text':
            out = text_src_seq
        elif self.args.modality == 'eeg':
            out = eeg_src_seq
            

        print(out.shape)
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        if self.use_bn:
            out = self.final_bn(out)
        print(out.shape)
        out = self.final_relu(out)
        print(out.shape)
        # out = out.mean(-1)
        if self.verbose:
            print('final pooling', out.shape)

        out = self.dense(out)
        if self.verbose:
            print('dense', out.shape)

        if self.verbose:
            print('softmax', out.shape)
        
        print(out.shape)
        return out    