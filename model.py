import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import loss_and_metrics


class TransformerEncoder(nn.Module):
    def __init__(self, config, num_rna_type, num_species, rbp_dim):
        super().__init__()
        self.config = config

        self.emb_dim = config["emb_dim"]     
        self.model_dim = config["model_dim"]
        self.num_heads = config["num_heads"]
        self.num_layers = config["num_layers"]
        self.num_labels = config["num_labels"]
        self.dropout = config["dropout"]

        # RNA_Type / Species embedding
        self.rna_type_proj = nn.Linear(num_rna_type, self.emb_dim)
        self.species_proj = nn.Linear(num_species, self.emb_dim)

        # RBP matrix → emb_dim
        self.rbp_proj = nn.Linear(rbp_dim, self.emb_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=self.num_heads,
            dim_feedforward=self.model_dim * 2,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.classifier = nn.Linear(self.emb_dim, self.num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x, rna_type_tensor, species_tensor, rbp_tensor):
        """
        x:          [B, L, emb_dim]        RNA embeddings
        rbp_tensor: [B, L, rbp_dim] or None or list  0/1 matrix (None if not available, or list)
        """
        B, L, D = x.size()

        if rbp_tensor is not None:
            if isinstance(rbp_tensor, list):
                rbp_processed = []
                for i, rbp in enumerate(rbp_tensor):
                    if rbp is None:
                        rbp_processed.append(x[i])
                    else:
                        rbp_proj = self.rbp_proj(rbp.float())  # [L_rbp, emb_dim]
                        L_rbp = rbp_proj.size(0)
                        
                        if L_rbp != L:
                            if L_rbp == 1:
                                rbp_proj = rbp_proj.expand(L, -1)
                            else:
                                # resize: [L_rbp, emb_dim] -> [1, emb_dim, L_rbp]
                                rbp_proj = rbp_proj.t().unsqueeze(0)  # [1, emb_dim, L_rbp]
                                rbp_proj = F.interpolate(rbp_proj, size=L, mode='linear', align_corners=False)
                                rbp_proj = rbp_proj.squeeze(0).t()  # [L, emb_dim]
                        
                        # element-wise multiplication
                        rbp_processed.append(x[i] * rbp_proj)
                
                x = torch.stack(rbp_processed)

        # CLS token
        cls_tokens = self.cls_token.expand(B, 1, D)  # [B, 1, emb_dim]

        # Category embeddings [B, *, emb_dim]
        rna_type_embed = self.rna_type_proj(rna_type_tensor).unsqueeze(1)
        species_embed = self.species_proj(species_tensor).unsqueeze(1)

        # Input sequence: [CLS, RNA_Type, Species, x_1...x_L]
        x = torch.cat([cls_tokens, rna_type_embed, species_embed, x], dim=1)

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # Extract CLS token only
        cls_token = x[:, 0, :]                  # [B, emb_dim]
        logits = self.classifier(cls_token)     # [B, num_labels]
        return logits

    def compute_loss_and_metrics(self, x, rna_type_tensor, species_tensor, rbp_tensor, y_true):
        logits = self.forward(x, rna_type_tensor, species_tensor, rbp_tensor)
        return loss_and_metrics(logits, y_true, self.loss_fn)
