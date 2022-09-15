import torch
from torch_geometric.nn import Linear, HeteroConv, GraphConv
from torch_geometric.nn.models import JumpingKnowledge

class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, labeled_class='item', hidden_channels=512, 
                 out_channels=2, n_layers=2):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.lin = Linear(-1, out_channels)        
        
        for i in range(n_layers):
            second_channels = hidden_channels if i < n_layers - 1 else out_channels
            
            conv = HeteroConv({
                edge_type: GraphConv(-1, second_channels)
                for edge_type in metadata[1]
            })
            if i < n_layers - 1:
                bn = torch.nn.ModuleDict(
                    {
                        node_type: torch.nn.BatchNorm1d(second_channels)
                        for node_type in metadata[0]
                    }
                )
                self.bns.append(bn)
            self.convs.append(conv)

        self.jk = JumpingKnowledge("cat")
        self.dropout = torch.nn.Dropout(0.5)
        self.activation = torch.nn.ReLU(inplace=True)
        self.emb = torch.nn.Embedding(3, 256)
        torch.nn.init.xavier_normal_(self.emb.weight)
        self.labeled_class = labeled_class

    def forward(self, x_dict, edge_index_dict, y_emb=None):

        labeled_class = self.labeled_class

        if y_emb is not None:
            y_emb = self.dropout(
                self.emb(y_emb) * (y_emb != 2).float().unsqueeze(1))
            x_dict[labeled_class] = x_dict[labeled_class] + y_emb

        if self.training:
            x_dict = {key: self.add_noise(x) for key, x in x_dict.items()}

        xs = [x_dict[labeled_class]]
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < len(self.convs) - 1:
                x_dict = {
                    key: self.dropout(self.activation(self.bns[i][key](x)))
                    for key, x in x_dict.items()
                }
            xs.append(x_dict[labeled_class])

        x = self.lin(self.jk(xs))

        return x

    @staticmethod
    def add_noise(x, perturb_noise=0.05):
        perturb = torch.empty_like(x).uniform_(-perturb_noise, perturb_noise)
        return x + perturb
