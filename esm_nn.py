
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, roc_auc_score

import matplotlib
from torch import nn, optim

from esm_embeddings import get_esm_sequence_embedding, load_peptide_data, get_esm_model, plot_roc_curve

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
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



chosen_embedding_size = 1280
chosen_embedding_layer = 33
chosen_test_size = 0.1
TRAIN = True

print("Loading peptide data")
positive_pep, negative_pep, doubt_lables,per_hash = load_peptide_data()

print("Loading ESM-2 model")
model_esm, alphabet_esm, batch_converter_esm, device_esm = get_esm_model(embedding_size=chosen_embedding_size)

print("Getting the ESM-2 embeddings for all the peptide data")
# Flatten token-level embeddings for training
def flatten_embeddings_and_labels(sequence_representations, labels):
    flattened = []
    flattened_labels = []
    for seq, label in zip(sequence_representations, labels):
        flattened.extend(seq)
        flattened_labels.extend([label] * len(seq))
    return np.array(flattened), np.array(flattened_labels)

positive_esm_emb = get_esm_sequence_embedding(positive_pep, model_esm, alphabet_esm, batch_converter_esm, device_esm, embedding_layer=chosen_embedding_layer,per_amino_acid=True)
negative_esm_emb = get_esm_sequence_embedding(negative_pep, model_esm, alphabet_esm, batch_converter_esm, device_esm, embedding_layer=chosen_embedding_layer,per_amino_acid=True)

all_embeddings = positive_esm_emb + negative_esm_emb
all_labels = [1] * len(positive_esm_emb) + [0] * len(negative_esm_emb)

X, y = flatten_embeddings_and_labels(all_embeddings, all_labels)



X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(X, y, range(len(X)), test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)



input_size = X_train.shape[1]
hidden_size = 1024
output_size = 1

model = FeedforwardNN(input_size, hidden_size, output_size)
if TRAIN:

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
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



    torch.save(model.state_dict(), 'model_nn.pth')
else:
    model.load_state_dict(torch.load('model_nn.pth'))


model.eval()

with torch.no_grad():
    y_pred_prob = model(X_test_tensor).numpy()
    y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate AUC
auc = roc_auc_score(y_test, y_pred_prob)
print(f'AUC: {auc:.4f}')
y_proba = model(X_test_tensor).detach().numpy()
plot_roc_curve(y_test, y_proba, f"roc_curve full_tokens.png")

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


positive_pep_new = [ seq[1] for seq in positive_pep]
index_array = []
flattened_positive_pep = []
for i in range(len(positive_pep_new)):
    for j in range(len(positive_pep_new[i])):
        flattened_positive_pep.append(positive_pep_new[i][j])
        index_array.append((i, j))

original_positive_indices = np.where(y_test== 1)[0]
positive_pep_test_indices = np.array(test_idx)[original_positive_indices]
positive_pep_test = np.array(flattened_positive_pep)[positive_pep_test_indices]
positive_test_embeddings = [X[idx] for idx in positive_pep_test_indices]

positive_test_embeddings_flattened, _ = flatten_embeddings_and_labels(positive_test_embeddings, [1] * len(positive_test_embeddings))


def compute_saliency_map(model, X, target_class=0):
    model.eval()
    X_var = torch.tensor(X, requires_grad=True)
    output = model(X_var)

    # Select the target class and compute gradients
    target_output = output[:, target_class]
    model.zero_grad()
    target_output.backward(torch.ones_like(target_output))

    # Aggregate the saliency map by taking the sum of absolute values of gradients across the embedding dimension
    saliency = X_var.grad.abs().sum(dim=-1).cpu().numpy()
    return saliency


# Compute saliency maps for the positive test embeddings
saliency_maps = compute_saliency_map(model, positive_test_embeddings)
import matplotlib.pyplot as plt



def plot_mean_saliency_map(sequences, saliency_maps):
    amino_acid_saliency = {}
    amino_acid_count = {}

    for aa, saliency in zip(sequences, saliency_maps):
        if aa not in amino_acid_saliency:
                amino_acid_saliency[aa] = 0
                amino_acid_count[aa] = 0
        else:
            amino_acid_saliency[aa] += saliency
            amino_acid_count[aa] += 1

    mean_saliency = {aa: amino_acid_saliency[aa] / amino_acid_count[aa] for aa in amino_acid_saliency}

    amino_acids = list(mean_saliency.keys())
    mean_saliency_values = list(mean_saliency.values())

    plt.figure(figsize=(12, 3))
    plt.bar(amino_acids, mean_saliency_values, color='blue')
    plt.title('Mean Saliency Map for All Sequences')
    plt.xlabel('Amino Acid')
    plt.ylabel('Mean Saliency')
    plt.show()





def plot_saliency_map(sequence, saliency_map):
    plt.figure(figsize=(12, 3))
    plt.bar(range(len(sequence)), saliency_map[:len(sequence)], color='blue')
    plt.xticks(ticks=np.arange(len(sequence)), labels=list(sequence), rotation=90)
    plt.title('Saliency Map')
    plt.xlabel('Amino Acid Position')
    plt.ylabel('Saliency')
    plt.show()


positive_pep_test_seq = positive_pep_test[:20]
plot_saliency_map(positive_pep_test_seq, saliency_maps[:len(positive_pep_test_seq)])
plot_mean_saliency_map(positive_pep_test,saliency_maps)

def load_model(model_path, input_size, hidden_size, output_size):
    model = FeedforwardNN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def predict(model, embedding):
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(embedding_tensor)
    return output.item()
