import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adjacency_matrix):
        degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=-1))
        normalized_adjacency = torch.matmul(torch.inverse(degree_matrix), adjacency_matrix)
        output = torch.matmul(torch.matmul(normalized_adjacency, x), self.weight)
        return output

class GCNModel(nn.Module):
    def __init__(self, input_features, hidden_features, output_classes):
        super(GCNModel, self).__init__()
        self.conv1 = GraphConvLayer(input_features, hidden_features)
        self.conv2 = GraphConvLayer(hidden_features, output_classes)

    def forward(self, x, adjacency_matrix):
        x = F.relu(self.conv1(x, adjacency_matrix))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adjacency_matrix)
        return x
