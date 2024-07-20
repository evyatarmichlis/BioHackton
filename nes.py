import sys
import torch
import torch.nn as nn
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from esm_embeddings import get_esm_model, get_esm_sequence_embedding
from torch.nn.utils.rnn import pad_sequence

# # Check command line arguments
if len(sys.argv) < 2:
    print("Usage: python nes.py <sequence> or python nes.py <sequence> plot ")
    sys.exit(1)

# Get the sequence from the command-line arguments
sequence = sys.argv[1]

# Check if the plot flag is provided
plot_flag = False
if len(sys.argv) == 3 and sys.argv[2].lower() == "plot":
    plot_flag = True

# Print the sequence (or perform any other operations with it)
print(f"You entered the sequence: {sequence}")


# Check if the sequence length is sufficient
class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

def load_model(model_path, input_size, hidden_size, output_size):
    model = FeedforwardNN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def get_esm_embedding(sequence, model_esm, alphabet, batch_converter, device):
    data = [("seq1", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = model_esm(batch_tokens.to(device), repr_layers=[33])
    token_representations = results["representations"][33]
    sequence_representation = token_representations[0, 1: len(sequence) + 1].cpu().numpy()

    return sequence_representation


def predict(model, embedding):
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(embedding_tensor)
    return output.item()


def compute_saliency_map(model, X, target_class=0):
    model.eval()
    X_var = torch.tensor(X, requires_grad=True)
    output = model(X_var)
    target_output = output[:, target_class]
    model.zero_grad()
    target_output.backward(torch.ones_like(target_output))
    saliency = X_var.grad.abs().sum(dim=-1).numpy()
    return saliency
# Define the RNN model class
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size_rnn, num_layers_rnn, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h_0 = torch.zeros(num_layers_rnn, x.size(0), hidden_size_rnn).to(device)
        c_0 = torch.zeros(num_layers_rnn, x.size(0), hidden_size_rnn).to(device)

        out, _ = self.rnn(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

# Load the trained model
input_size_rnn = 1280  # Set the correct input size
hidden_size_rnn = 128
num_layers_rnn = 2
output_size_rnn = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNNModel(input_size_rnn, hidden_size_rnn, num_layers_rnn, output_size_rnn).to(device)
model.load_state_dict(torch.load('model_rnn.pth'))
model.eval()

# Get the ESM model and other related components
chosen_embedding_size = 1280
chosen_embedding_layer = 33
model_esm, alphabet_esm, batch_converter_esm, device_esm = get_esm_model(embedding_size=chosen_embedding_size)
if plot_flag:
    input_size_nn = 1280
    hidden_size_nn = 1024
    output_size_nn = 1
    model_nn = FeedforwardNN(input_size_nn,hidden_size_nn,output_size_nn)
def integrated_gradients(model, X, baseline=None, steps=50):
    if baseline is None:
        baseline = torch.zeros_like(X).to(device)
    assert baseline.shape == X.shape

    # Scale inputs and compute gradients
    scaled_inputs = [baseline + (float(i) / steps) * (X - baseline) for i in range(0, steps + 1)]
    grads = []
    for scaled_input in scaled_inputs:
        scaled_input.requires_grad = True
        output = model(scaled_input)
        target_output = output[:, 0]  # Assuming binary classification
        model.zero_grad()
        target_output.backward(retain_graph=True)
        grads.append(scaled_input.grad.detach().cpu().numpy())

    avg_grads = np.mean(grads, axis=0)
    integrated_grad = (X.cpu().detach().numpy() - baseline.cpu().detach().numpy()) * avg_grads
    return integrated_grad



def compute_saliency_map(model, X, target_class=0):
    model.eval()
    X_var = X.clone().detach().requires_grad_(True).to(device)
    output = model(X_var.unsqueeze(0))

    # Select the target class and compute gradients
    target_output = output[:, target_class]
    model.zero_grad()
    target_output.backward(torch.ones_like(target_output))

    # Aggregate the saliency map by taking the sum of absolute values of gradients across the embedding dimension
    saliency = X_var.grad.abs().sum(dim=-1).cpu().numpy()
    return saliency

def plot_saliency_map(sequence, saliency_map):
    plt.figure(figsize=(12, 3))
    plt.bar(range(len(sequence)), np.squeeze(saliency_map)[:len(sequence)], color='blue')
    plt.xticks(ticks=np.arange(len(sequence)), labels=list(sequence), rotation=90)
    plt.title('Saliency Map')
    plt.xlabel('Amino Acid Position')
    plt.ylabel('Saliency')
    plt.show()



def pad_single_sequence(sequence_embedding, min_length=22, padding_value=0.0):
    seq_tensor = torch.tensor(sequence_embedding)
    if seq_tensor.size(0) < min_length:
        padding = torch.full((min_length - seq_tensor.size(0), seq_tensor.size(1)), padding_value)
        seq_tensor = torch.cat((seq_tensor, padding), dim=0)
    return seq_tensor
# Function to run the sliding window and make predictions
def run_sliding_window(sequence, window_size=min(22, len(sequence))):
    windows = [(i, sequence[i:i + window_size]) for i in range(len(sequence) - window_size + 1)]
    embeddings = get_esm_sequence_embedding(windows, model_esm, alphabet_esm, batch_converter_esm, device_esm, embedding_layer=chosen_embedding_layer, per_amino_acid=True)
    for (start_idx, window), emb in zip(windows, embeddings):
        emb = pad_single_sequence(emb)
        emb_tensor = torch.tensor(emb).unsqueeze(0).to(device)

        with torch.no_grad():

            prediction = model(emb_tensor).item()
            label = int(prediction > 0.5)
            print(f"Window starting at index {start_idx}: Predicted label = {label}")


def run_nn(seq):
    embedding = get_esm_sequence_embedding([(0,seq)], model_esm, alphabet_esm, batch_converter_esm, device_esm, embedding_layer=chosen_embedding_layer, per_amino_acid=True)
    model_nn.load_state_dict(torch.load('model_nn.pth'))
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
    saliency_map = compute_saliency_map(model_nn, embedding_tensor)
    saliency_map = np.squeeze(saliency_map)
    plot_saliency_map(sequence, saliency_map)



run_sliding_window(sequence)
if plot_flag:
    run_nn(sequence)


