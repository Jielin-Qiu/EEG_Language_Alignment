import torch
import torch.nn as nn
from block_new import get_sinusoid_encoding_table, get_attn_key_pad_mask, get_non_pad_mask, \
    get_subsequent_mask, EncoderLayer, EncoderLayer2, EncoderLayer3
from config import PAD, KS, Fea_PLUS
import torch.nn.functional as F
from CCA import cca_loss
from torch.nn.modules.batchnorm import BatchNorm1d


class Encoder(nn.Module):


    def __init__(
            self,
            d_feature,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):
        super().__init__()

        n_position = d_feature + 1
        self.src_word_emb = nn.Conv1d(1, d_model, kernel_size=KS, padding=int((KS - 1) / 2))

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos):

        non_pad_mask = get_non_pad_mask(src_seq)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        enc_output = src_seq.unsqueeze(1)
        enc_output = self.src_word_emb(enc_output)
        enc_output = enc_output.transpose(1, 2)
        enc_output.add_(self.position_enc(src_pos))

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output,


class Encoder2(nn.Module):


    def __init__(
            self,
            d_feature,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):
        super().__init__()

        n_position = d_feature + 1
        self.src_word_emb = nn.Conv1d(1, d_model, kernel_size=KS, padding=int((KS - 1) / 2))

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer2(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos):

        non_pad_mask = get_non_pad_mask(src_seq)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        enc_output = src_seq.unsqueeze(1)
        enc_output = self.src_word_emb(enc_output)
        enc_output = enc_output.transpose(1, 2)
        enc_output.add_(self.position_enc(src_pos))

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output,


class Encoder3(nn.Module):


    def __init__(
            self,
            d_feature,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):
        super().__init__()

        n_position = d_feature + 1
        self.src_word_emb = nn.Conv1d(1, d_model, kernel_size=KS, padding=int((KS - 1) / 2))

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer3(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos):

        non_pad_mask = get_non_pad_mask(src_seq)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        enc_output = src_seq.unsqueeze(1)
        enc_output = self.src_word_emb(enc_output)
        enc_output = enc_output.transpose(1, 2)
        enc_output.add_(self.position_enc(src_pos))

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''
    def __init__(
            self, device,
            d_feature, d_model, d_inner,
            n_layers, n_head, d_k=64, d_v=64, dropout = 0.1,
            class_num=2):

        super().__init__()

        self.encoder = Encoder(d_feature, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)
        self.device = device

        self.linear1_cov = nn.Conv1d(d_feature, 1, kernel_size=1)
        self.linear1_linear = nn.Linear(d_model, class_num)
        self.linear2_cov = nn.Conv1d(d_model, 1, kernel_size=1)
        self.linear2_linear = nn.Linear(d_feature, class_num)
        self.bn=nn.BatchNorm1d(d_feature)
        self.bn2=nn.BatchNorm1d(16) 
        self.dropout = nn.Dropout(0.1)

    def forward(self, src_seq):
        b, l = src_seq.size()
     
        src_pos = torch.LongTensor(
            [list(range(1, l + 1)) for i in range(b)]
        )
        src_pos = src_pos.to(self.device)

        enc_output, *_ = self.encoder(src_seq, src_pos)
        
        res = self.linear1_cov(enc_output)
        res = res.contiguous().view(res.size()[0], -1)
        res = self.linear1_linear(res)
        return res


class Transformer2(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''
    def __init__(
            self, device,
            d_feature, d_model, d_inner,
            n_layers, n_head, d_k=64, d_v=64, dropout = 0.1,
            class_num=2):

        super().__init__()

        self.encoder = Encoder2(d_feature, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)
        self.device = device

        self.linear1_cov = nn.Conv1d(d_feature, 1, kernel_size=1)
        self.linear1_linear = nn.Linear(d_model, class_num)
        self.linear2_cov = nn.Conv1d(d_model, 1, kernel_size=1)
        self.linear2_linear = nn.Linear(d_feature, class_num)
        self.bn=nn.BatchNorm1d(d_feature)
        self.bn2=nn.BatchNorm1d(16) 
        self.dropout = nn.Dropout(0.2)

    def forward(self, src_seq):
        b, l = src_seq.size()
     
        src_pos = torch.LongTensor(
            [list(range(1, l + 1)) for i in range(b)]
        )
        src_pos = src_pos.to(self.device)

        enc_output, *_ = self.encoder(src_seq, src_pos)
        
        res = self.linear1_cov(enc_output)
        res = res.contiguous().view(res.size()[0], -1)

        res = self.linear1_linear(res)
        return res


class Transformer3(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''
    def __init__(
            self, device,
            d_feature, d_model, d_inner,
            n_layers, n_head, d_k=64, d_v=64, dropout = 0.1,
            class_num=2):

        super().__init__()

        self.encoder = Encoder3(d_feature, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)
        self.device = device

        self.linear1_cov = nn.Conv1d(d_feature, 1, kernel_size=1)
        self.linear1_linear = nn.Linear(d_model, class_num)
        self.linear2_cov = nn.Conv1d(d_model, 1, kernel_size=1)
        self.linear2_linear = nn.Linear(d_feature, class_num)
        self.bn=nn.BatchNorm1d(d_feature)
        self.bn2=nn.BatchNorm1d(16) #batch
        self.dropout = nn.Dropout(0.2)

    def forward(self, src_seq):
        b, l = src_seq.size()
     
        src_pos = torch.LongTensor(
            [list(range(1, l + 1)) for i in range(b)]
        )
        src_pos = src_pos.to(self.device)

        enc_output, *_ = self.encoder(src_seq, src_pos)
        
        res = self.linear1_cov(enc_output)
        res = res.contiguous().view(res.size()[0], -1)

        res = self.linear1_linear(res)
        return res


class DeepCCA(nn.Module):
    def __init__(self, model1, model2, outdim_size, use_all_singular_values, device=torch.device('cuda')):
        super(DeepCCA, self).__init__()
        self.model1 = model1
        self.model2 = model2
       
        self.loss = cca_loss(outdim_size, use_all_singular_values, device).loss

    def forward(self, x1, x2):
        
        output1 = self.model1(x1)
        output2 = self.model2(x2)
        
        

        return output1, output2


class DeepCCA_fusion(nn.Module):
    def __init__(self, model1, outdim_size, use_all_singular_values, d_feature, d_model, d_inner,
            n_layers, n_head, d_k=64, d_v=64, dropout = 0.5,
            class_num=2, device=torch.device('cuda')):
        super(DeepCCA_fusion, self).__init__()
        self.model1 = model1
        self.Transformer = Transformer3(device=device, d_feature=4, d_model=d_model, d_inner=d_inner,
                            n_layers=n_layers, n_head=n_head, d_k=64, d_v=64, dropout=dropout, class_num=class_num)

        self.loss = cca_loss(outdim_size, use_all_singular_values, device).loss
        self.classifier = nn.Linear(4, class_num)

    def forward(self, x1, x2):
        
        x1, x2 = self.model1(x1, x2)
        x = torch.cat((x1, x2), dim = 1)
        out = self.classifier(F.relu(x))

        return out, x1, x2


class Fusion(nn.Module):
  def __init__(self, device, model1, model2,
            d_feature, d_model, d_inner,
            n_layers, n_head, d_k=64, d_v=64, dropout = 0.5,
            class_num=3):
    super(Fusion, self).__init__()
    self.device = device
    self.model1 = model1
    self.model2 = model2
    self.Transformer = Transformer3(device=device, d_feature=4, d_model=d_model, d_inner=d_inner,
                            n_layers=n_layers, n_head=n_head, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
    self.classifier = nn.Linear(4, class_num)
    self.bn = nn.BatchNorm1d(4)
    self.dropout = nn.Dropout(0.25)

  def forward(self, x1, x2):
    x1 = self.model1(x1)
    x2 = self.model2(x2)
    
    x = torch.cat((x1, x2), dim = 1)
    out = self.classifier(x)

    return out, x1, x2


class CAM(nn.Module):
    def __init__(self, model1, model2):
        super(Fusion, self).__init__()
        self.model1 = model1
        self.model2 = model2       

    def forward(self, x1, x2):
        
        output1 = self.model1(x1)
        output2 = self.model2(x2)

        return output1, output2


class BiLSTM(nn.Module):
    #vocab_size = 48, 32, 838
    def __init__(self, vocab_size, device, embedding_dim = 32, hidden_dim1 = 128, hidden_dim2 = 64, output_dim = 3, n_layers =2,
                 dropout = 0.3, bidirectional = True, pad_index = 1):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_index)
        
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim1,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
        self.fc1 = nn.Linear(hidden_dim1 * 2, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        b, l = text.size()
        text_lengths = torch.tensor([32]*b).cpu()
        src_pos = torch.LongTensor(
            [list(range(0, l)) for i in range(b)]
        )
        src_pos = src_pos.to(self.device)

        embedded = self.embedding(src_pos)

        packed_output, (hidden, cell) = self.lstm(text)

        rel = self.relu(packed_output)
        dense1 = self.fc1(rel)
        drop = self.dropout(dense1)
        preds = self.fc2(drop)
        return preds

class MLP(nn.Module):
  #vocab_size = 48, 32, 838
    def __init__(self, vocab_size, embed_size = 1, hidden_size2 = 256, hidden_size3 = 128, hidden_size4 = 64, 
    output_dim = class_num, dropout = 0.3, max_document_length = 32):
        super().__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_size*max_document_length, hidden_size2)  
        self.fc2 = nn.Linear(hidden_size2, hidden_size3) 
        self.fc3 = nn.Linear(hidden_size3, hidden_size4)  
        self.fc4 = nn.Linear(hidden_size4, output_dim)  

    def forward(self, text):
 
        x = self.relu(self.fc1(text))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        preds = self.fc4(x)
        return preds

class MLP2(nn.Module):
  #vocab_size = 48, 32, 838
    def __init__(self, vocab_size, embed_size = 1, hidden_size2 = 256, hidden_size3 = 128, hidden_size4 = 64, 
    output_dim = class_num, dropout = 0.3, max_document_length = 838):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_size*max_document_length, hidden_size2)  
        self.fc2 = nn.Linear(hidden_size2, hidden_size3)  
        self.fc3 = nn.Linear(hidden_size3, hidden_size4)  
        self.fc4 = nn.Linear(hidden_size4, output_dim)  

    def forward(self, text):

        x = self.relu(self.fc1(text))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        preds = self.fc4(x)
        return preds


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

    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, verbose=False):
        super(ResNet1D, self).__init__()
        
        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap 
        self.increasefilter_gap = increasefilter_gap 

        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
                
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
                in_channels = base_filters
                out_channels = in_channels
            else:

                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
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

        
    def forward(self, x):
        
        out = x

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
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.verbose:
            print('final pooling', out.shape)

        out = self.dense(out)
        if self.verbose:
            print('dense', out.shape)

        if self.verbose:
            print('softmax', out.shape)
        
        return out    