import torch
from torch import nn
from .BaseModel import BaseModel2

class Dynaformer(BaseModel2):
    def __init__(self, final_out_dim=1,hidden_dim=128, lr=1e-3, is_instance_norm=True, loss="mse", patience_lr_plateau=100):
        super(Dynaformer, self).__init__()
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
        pre_encoder =  self.pre_encoder_linear(context_value)


        positional_encoding = self.position_embedding_context(time.long())
        encoder = pre_encoder + positional_encoding #torch.cat([pre_encoder,positional_encoding], axis=2)
        
        encoder = self.encoder_transformer(encoder.permute(1,0,2), src_key_padding_mask=src_padding_mask.bool())
        return encoder, src_padding_mask

    def forward(self, context, x):
        encoder, src_padding_mask = self.embedding(context)
        x_padded = torch.nn.functional.pad(x, (0,  self.chunks  - (x.shape[1] % self.chunks), 0 ,0))
        query = x_padded.reshape(x_padded.shape[0], x_padded.shape[1] // self.chunks , self.chunks)
        collate_mask = torch.zeros(query.shape[:-1], device=self.device)
        collate_mask[(query == 0).all(axis=2)]=1.
        all_query = self.step_up_query(query)
 
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

