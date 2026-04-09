import torch
import torch.nn as nn

class QorrectTransformer(nn.Module):
    def __init__(self, input_dim=17, d_model=128, nhead=8, num_layers=4, output_dim=16):
        # input_dim = 16 (noisy probs) + 1 (noise level)
        super().__init__()

        # Project input into transformer dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
            nn.Softmax(dim=-1)  # output is a probability distribution
        )

    def forward(self, x):
        # x shape: (batch, 17) → project and add sequence dim
        x = self.input_proj(x).unsqueeze(1)  # (batch, 1, d_model)
        x = self.transformer(x)
        x = x.squeeze(1)                      # (batch, d_model)
        return self.output_head(x)


if __name__ == "__main__":
    import torch
    model = QorrectTransformer()
    dummy = torch.randn(8, 17)  # batch of 8
    out = model(dummy)
    print("✅ Model output shape:", out.shape)
    print("✅ Sample probabilities sum:", out[0].sum().item())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Total parameters: {total_params:,}")