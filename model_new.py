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
