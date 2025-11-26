import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# ====== 1. Load dataset ======
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# ====== 2. Better GCN model ======
class BetterGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 1st layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 2nd layer
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

model = BetterGCN(
    in_channels=data.num_node_features,
    hidden_channels=16,             
    out_channels=dataset.num_classes,
    dropout=0.5
).to(device)

# ====== 3. Optimizer with weight decay ======
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=5e-4      
)

# ====== 4. Helper: accuracy ======
@torch.no_grad()
def evaluate():
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)

    accs = {}
    for split, mask in [('train', data.train_mask),
                        ('val', data.val_mask),
                        ('test', data.test_mask)]:
        correct = (pred[mask] == data.y[mask]).sum()
        accs[split] = int(correct) / int(mask.sum())
    return accs

# ====== 5. Training loop ======
best_val_acc = 0
best_test_acc = 0

for epoch in range(200):   
    model.train()
    optimizer.zero_grad()

    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    accs = evaluate()
    train_acc, val_acc, test_acc = accs['train'], accs['val'], accs['test']

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc

    if (epoch + 1) % 10 == 0:
        print(
            f'Epoch: {epoch+1:03d}, '
            f'Loss: {loss:.4f}, '
            f'Train: {train_acc:.3f}, Val: {val_acc:.3f}, Test: {test_acc:.3f}'
        )

print(f'Best Val Acc: {best_val_acc:.3f}, Corresponding Test Acc: {best_test_acc:.3f}')
