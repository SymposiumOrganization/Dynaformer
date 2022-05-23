import torch
from torch import nn
import pdb

from .BaseModel import BaseModel,BaseModel2
import torch.optim as optim
import numpy as np


class QREncoder(nn.Module):
    def __init__(self, in_channels, out_dimension, sig_depth, stream):
        super(QREncoder, self).__init__()
        self.augment = signatory.Augment(in_channels=in_channels,
                                         layer_sizes=(),
                                         kernel_size=1,
                                         include_original=True,
                                         include_time=True)
        self.sig_depth=sig_depth
        self.signature = signatory.Signature(depth=sig_depth,stream=stream)
        sig_channels = signatory.signature_channels(channels=in_channels + 1,
                                                    depth=sig_depth)
        self.linear = torch.nn.Linear(sig_channels,
                                      out_dimension)      
    def forward(self, inp):
        x = self.augment(inp)
        y=self.signature(x)
        z = self.linear(y)
        return z


class VIDecoder(nn.Module):
    def __init__(self, in_channels, out_dimension, sig_depth, stream):
        super(VIDecoder, self).__init__()
        self.augment = signatory.Augment(in_channels=in_channels,
                                         layer_sizes=(),
                                         kernel_size=1,
                                         include_original=True,
                                         include_time=True)
        self.sig_depth=sig_depth
        self.signature = signatory.Signature(depth=sig_depth,stream=stream)
        sig_channels = signatory.signature_channels(channels=in_channels + 1,
                                                    depth=sig_depth)
        self.linear = torch.nn.Linear(sig_channels,
                                      out_dimension)

    def forward(self, inp):
        x = self.augment(inp)
        y=self.signature(x)
        y=(torch.cat([torch.zeros((y.shape[0],1,y.shape[2]),device=y.device),y],axis=1))
        z = self.linear(y)
        return z


class Conditional(BaseModel2):
    def __init__(self, in_channels_e=2,in_channels_d=1, out_dimension=100, final_out_dim=1, sig_depth=3, norm=False,lr=1e-3):
        super(Conditional, self).__init__()
        self.encoder=QREncoder(in_channels_e,out_dimension,sig_depth,False)
        self.decoder=VIDecoder(in_channels_d,out_dimension,sig_depth,True)
        self.linear = torch.nn.Linear(out_dimension, final_out_dim)
        self.out_dimension=out_dimension
        self.norm=norm 
        self.lr = lr  
        self.loss_func = torch.nn.MSELoss(reduction='mean')
        self.save_hyperparameters()

    def forward(self, context, x):
        a=self.encoder(context).unsqueeze(1)
        b=self.decoder(x)
        if self.norm:
          c=torch.einsum('ijk,ilk->ilk',a,b)/np.sqrt(self.out_dimension)
        else:
          c=torch.einsum('ijk,ilk->ilk',a,b)
        return self.linear(c)

def masked_instance_norm(x, mask, eps = 1e-5):
    """
    x of shape: [batch_size (N), num_objects (L), features(C)]
    mask of shape: [batch_size (N), num_objects (L)]
    """
    mask = mask.float().unsqueeze(-1)  # (N,L,1)
    mean = (torch.sum(x * mask, 1) / torch.sum(mask, 1))   # (N,C)
    mean = mean.detach()
    var_term = ((x - mean.unsqueeze(1).expand_as(x)) * mask)**2  # (N,L,C)
    var = (torch.sum(var_term, 1) / torch.sum(mask, 1))  #(N,C)
    var = var.detach()
    mean_reshaped = mean.unsqueeze(1).expand_as(x)  # (N, L, C)
    var_reshaped = var.unsqueeze(1).expand_as(x)    # (N, L, C)
    ins_norm = (x - mean_reshaped) / torch.sqrt(var_reshaped + eps)   # (N, L, C)
    return ins_norm

class TransfomerBasedArchitecture(BaseModel2):
    def __init__(self, final_out_dim=1,hidden_dim=128, lr=1e-3, is_instance_norm=True, loss="mse", patience_lr_plateau=100):
        super(TransfomerBasedArchitecture, self).__init__()
        # # Define a standard cnn layer as encoder #DEPRECATED, does not work properly batch norm
        self.pre_encoder_conv = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, stride=1, padding=1),
            #nn.InstanceNorm1d(16, affine=True),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            #nn.InstanceNorm1d(32, affine=True),+
            nn.ReLU(),
            nn.Conv1d(32, hidden_dim, kernel_size=3, stride=1, padding=1),
            #nn.InstanceNorm1d(hidden_dim, affine=True),
            nn.ReLU())    
        self.pre_encoder_linear = nn.Linear(2, hidden_dim)
        #self.pre_encoder_linear = nn.Linear(2, hidden_dim)  
        self.step_up_query = nn.Sequential(nn.Linear(hidden_dim//2, hidden_dim))
        self.last_query = nn.Linear(hidden_dim//2, hidden_dim)
        self.is_instance_norm = is_instance_norm
        self.instance_norm_all_but_last_pre_query = nn.InstanceNorm1d(hidden_dim,  affine=False)
        self.hidden_dim = int(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.encoder_transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_dim , nhead=8)
        self.decoder=nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.position_embedding_context = nn.Embedding(1000, self.hidden_dim )
        self.positional_embedding_query = nn.Embedding(num_embeddings=250, embedding_dim=self.hidden_dim )
        self.linear = torch.nn.Linear(self.hidden_dim , final_out_dim)
        self.lr = lr  
        self.loss_func = torch.nn.MSELoss(reduction='mean')
        self.pre_fin = nn.Linear(self.hidden_dim, self.hidden_dim//2)
        self.chunks = self.hidden_dim//2
        self.loss = loss
        self.patience_lr_plateau = patience_lr_plateau
        self.save_hyperparameters()

    def embedding(self, context):
        context_value, time = context[:,:,:2], context[:,:,2]
        src_padding_mask = torch.zeros(context_value.shape[0], context_value.shape[1], device=context_value.device)
        src_padding_mask[(context_value == 0).all(axis=2)]=1
        #good_values = ~src_padding_mask.bool()
        #context_value[:,:,1:2] = masked_instance_norm(context_value[:,:,1:2], good_values)
        #all_query_normalized = context_value
        pre_encoder =  self.pre_encoder_linear(context_value)

        # Add positional encoding
        #time = torch.arange(time.shape[-1], device=self.device)
        positional_encoding = self.position_embedding_context(time.long())
        #positional_encoding = positional_encoding.unsqueeze(0).repeat(context.shape[0], 1, 1)
        encoder = pre_encoder + positional_encoding #torch.cat([pre_encoder,positional_encoding], axis=2)
        #encoder = pre_encoder
        #encoder_debug = encoder.clone().detach()
        encoder = self.encoder_transformer(encoder.permute(1,0,2), src_key_padding_mask=src_padding_mask.bool())
        return encoder, src_padding_mask

    def forward(self, context, x):
        encoder, src_padding_mask = self.embedding(context)

        #debug_val = self.encoder_transformer(encoder_debug.permute(1,0,2)[:,-1:,:], src_key_padding_mask=src_padding_mask.bool()[-1:,:])
        #breakpoint()
        #torch.isclose(encoder[:,-1:,:], debug_val,atol=0.0001)
        # reshape X into chunks of 64 elements each
        # first pad the x to make it a multiple of 64 in the second dimension (X has shape (batch, stream, 1))
        x_padded = torch.nn.functional.pad(x, (0,  self.chunks  - (x.shape[1] % self.chunks), 0 ,0))
        #bool_mask = torch.ones(x_padded.shape[0], x_padded.shape[1], x_padded.shape[2], device=self.device)
        #pad_sequence = self.chunks - (x.shape[1] % self.chunks)
        #bool_mask[:, -pad_sequence:, :] = 0
        query = x_padded.reshape(x_padded.shape[0], x_padded.shape[1] // self.chunks , self.chunks)
        collate_mask = torch.zeros(query.shape[:-1], device=self.device)
        collate_mask[(query == 0).all(axis=2)]=1.
        #bool_mask = bool_mask.reshape(bool_mask.shape[0], bool_mask.shape[1] // self.chunks , self.chunks)
        all_query = self.step_up_query(query)
        # Enlarge bool_mask to match the shape of all_but_last_query
        good_indeces = ~collate_mask.bool()
        if self.is_instance_norm:
            all_query_normalized = masked_instance_norm(all_query, good_indeces)
            query = all_query_normalized #torch.cat([all_but_last_query, last_query], dim=1)
        else:
            query = all_query
        positions = torch.arange(0,query.shape[1], device=self.device).unsqueeze(0).repeat(x.shape[0], 1)
        positional_embedding = self.positional_embedding_query(positions)
        # sum the positional embedding with the encoder output
        query = query + positional_embedding
        # now we have x.s  hape = (batch, stream // 128, 128), ready to be passed to the decoder
        # Prepare for the transfomer decoder
        #encoder = encoder.permute(1,0,2)  # seq_len, batch, hidden_dim
        query = query.permute(1,0,2) # seq_len, batch, hidden_dim
        output_transf = self.decoder(query,encoder,tgt_key_padding_mask=collate_mask.bool(), memory_key_padding_mask=src_padding_mask.bool())
        output_transf = output_transf.permute(1,0,2)
        out = self.pre_fin(output_transf)
        # restore the shape of the output
        out = out.reshape(out.shape[0], out.shape[1] * out.shape[2])
        assert out.shape[1] == x_padded.shape[1]
        # Remove the padding
        out = out[:, :x.shape[1]]
        return out.unsqueeze(2)


    


class S2S(BaseModel2):
    '''Seq2seq model.
    Input: [batch size, len_in, dim_in]
    Output: [batch size, len_out, dim_out]
    '''
    def __init__(self, dim_in=2, dim_out=1, hidden_size=100, cell='LSTM',patience_lr_plateau=100, lr=1e-3, loss="rmse"):
        super(S2S, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden_size = hidden_size
        self.cell = cell
        self.encoder = self.__init_encoder()
        self.decoder = self.__init_decoder()
        self.out = self.__init_out()
        self.lr=lr
        self.position_embedding_context = nn.Embedding(1000, dim_in - 2 )
        self.loss_func = torch.nn.MSELoss(reduction='mean')
        self.patience_lr_plateau = patience_lr_plateau
        self.loss = loss
        self.save_hyperparameters()
        
    def forward(self, context, x):
        x=x.unsqueeze(2)
        context_value, time = context[:,:,:2], context[:,:,2]
        time_emb = self.position_embedding_context(time.long())
        if len(x.size()) == 2:
            to_squeeze = True
        else:
            to_squeeze = False
        #to_squeeze = True if len(x.size()) == 2 else False
        almzero = torch.ones([1, x.size(0), self.hidden_size], dtype=x.dtype, device=x.device)/1e6
        if self.cell == "LSTM":
            init_state = (almzero, almzero) 
        else:
            almzero = almzero
        #init_state = (almzero, almzero) if self.cell == 'LSTM' else almzero
        context_value = torch.cat([context_value, time_emb], dim=2)
        output, other_stuff = self.encoder(context_value, init_state)
        h_n, c_n = other_stuff
        x, _ = self.decoder(x,  (h_n, c_n))
        x = self.out(x)
        return x.squeeze(0) if to_squeeze else x
        
    def __init_encoder(self):
        if self.cell == 'RNN':
            return torch.nn.RNN(self.dim_in, self.hidden_size, batch_first=True)
        elif self.cell == 'LSTM':
            return torch.nn.LSTM(self.dim_in, self.hidden_size, batch_first=True)
        elif self.cell == 'GRU':
            return torch.nn.GRU(self.dim_in, self.hidden_size, batch_first=True)
        else:
            raise NotImplementedError
    
    def __init_decoder(self):
        if self.cell == 'RNN':
            return torch.nn.RNN(1, self.hidden_size, batch_first=True)
        elif self.cell == 'LSTM':
            return torch.nn.LSTM(1, self.hidden_size, batch_first=True)
        elif self.cell == 'GRU':
            return torch.nn.GRU(1, self.hidden_size, batch_first=True)
        else:
            raise NotImplementedError
    
    def __init_out(self):
        return torch.nn.Linear(self.hidden_size, self.dim_out)