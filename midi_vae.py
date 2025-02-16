import torch
import torch.nn as nn
import torch.optim as optim

class MidiVAE(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, latent_size):
        super(MidiVAE, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        
        # Embedding layer to map token IDs into vectors.
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Encoder RNN: processes the sequence of embeddings.
        self.encoder_rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        
        # Fully connected layers to compute the latent mean and log variance.
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)
        
        # Decoder: map latent vector to hidden state and decode sequence.
        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)
        self.decoder_rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
    
    def encode(self, x, lengths):
        # x: [B, L] token sequence
        embedded = self.embedding(x)  # [B, L, embed_size]
        # Move lengths to CPU before packing
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder_rnn(packed)
        hidden = hidden[-1]  # [B, hidden_size]
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, seq_len):
        # Initialize the decoder's hidden state from the latent vector.
        hidden = self.latent_to_hidden(z).unsqueeze(0)  # [1, B, hidden_size]
        # For simplicity, we use zeros as inputs for every time step.
        batch_size = z.size(0)
        dec_input = torch.zeros(batch_size, seq_len, self.embedding.embedding_dim, device=z.device)
        dec_output, _ = self.decoder_rnn(dec_input, hidden)
        logits = self.output_layer(dec_output)  # [B, seq_len, vocab_size]
        return logits
    
    def forward(self, x, lengths):
        mu, logvar = self.encode(x, lengths)
        z = self.reparameterize(mu, logvar)
        seq_len = x.size(1)
        logits = self.decode(z, seq_len)
        return logits, mu, logvar
    
def vae_loss(logits, x, mu, logvar, pad_token=0):
    # Flatten logits and targets for cross-entropy loss.
    logits_flat = logits.view(-1, logits.size(-1))
    x_flat = x.view(-1)
    recon_loss = nn.CrossEntropyLoss(ignore_index=pad_token)(logits_flat, x_flat)
    # Compute KL divergence loss (averaged over the batch)
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return recon_loss + kl_loss

