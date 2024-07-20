import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from esm_embeddings import get_esm_sequence_embedding, load_peptide_data, get_esm_model
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# Function to pad sequences
def pad_sequences(sequences, padding_value=0.0):
    # Pad sequences to the same length
    padded_sequences = pad_sequence([torch.tensor(seq) for seq in sequences], batch_first=True, padding_value=padding_value)
    return padded_sequences
# Load peptide data and get ESM embeddings
chosen_embedding_size = 1280
chosen_embedding_layer = 33

print("Loading peptide data")
positive_pep, negative_pep, doubt_labels, per_hash = load_peptide_data()

print("Loading ESM-2 model")
model_esm, alphabet_esm, batch_converter_esm, device_esm = get_esm_model(embedding_size=chosen_embedding_size)

print("Getting the ESM-2 embeddings for all the peptide data")
positive_esm_emb = get_esm_sequence_embedding(positive_pep, model_esm, alphabet_esm, batch_converter_esm, device_esm, embedding_layer=chosen_embedding_layer,per_amino_acid=True)
negative_esm_emb = get_esm_sequence_embedding(negative_pep, model_esm, alphabet_esm, batch_converter_esm, device_esm, embedding_layer=chosen_embedding_layer,per_amino_acid=True)

# Combine embeddings and labels
all_embeddings = positive_esm_emb + negative_esm_emb
all_labels = [1] * len(positive_esm_emb) + [0] * len(negative_esm_emb)

# Pad the sequences
X = pad_sequences(all_embeddings)
y = torch.tensor(all_labels)

# Split the data
X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(X, y, range(len(X)), test_size=0.2, random_state=42)

TRAIN = False
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)

        out, _ = self.rnn(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


# Define model parameters
input_size = X_train.shape[2]
hidden_size = 128
num_layers = 2
output_size = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNNModel(input_size, hidden_size, num_layers, output_size).to(device)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
if TRAIN:
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'model_rnn.pth')
else:
    model.load_state_dict(torch.load('model_rnn.pth'))

model.eval()
with torch.no_grad():
    y_pred_prob = model(X_test_tensor).cpu().numpy()
    y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate AUC
auc = roc_auc_score(y_test.cpu(), y_pred_prob)
print(f'AUC: {auc:.4f}')

# Classification report and confusion matrix
print(classification_report(y_test.cpu(), y_pred))
print(confusion_matrix(y_test.cpu(), y_pred))

# Save the model's state dictionary

def compute_saliency_map(model, X, target_class=0):
    model.eval()
    X_var = X.clone().detach().requires_grad_(True).to(device)
    output = model(X_var)

    # Select the target class and compute gradients
    target_output = output[:, target_class]
    model.zero_grad()
    target_output.backward(torch.ones_like(target_output))

    # Aggregate the saliency map by taking the sum of absolute values of gradients across the embedding dimension
    saliency = X_var.grad.abs().sum(dim=-1).cpu().numpy()
    return saliency

def plot_saliency_map(sequence, saliency_map):
    plt.figure(figsize=(12, 3))
    plt.bar(range(len(sequence)[1]), np.squeeze(saliency_map)[:20][:len(sequence)], color='blue')
    plt.xticks(ticks=np.arange(len(sequence)), labels=list(sequence), rotation=90)
    plt.title('Saliency Map')
    plt.xlabel('Amino Acid Position')
    plt.ylabel('Saliency')
    plt.show()


positive_test_indices = (y_test == 1).nonzero(as_tuple=True)[0].tolist()
sequence_index = positive_test_indices[0]  # Take the first positive sequence

# Get the sequence and its saliency map
test_sequence = positive_pep[sequence_index]  # Original sequence
test_embedding = X_test_tensor[sequence_index].unsqueeze(0)  # Embedding

# Compute saliency map
saliency_map = compute_saliency_map(model, test_embedding)

# Plot saliency map
plot_saliency_map(test_sequence, saliency_map)