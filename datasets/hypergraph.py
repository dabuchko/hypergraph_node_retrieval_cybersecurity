import torch
from torch_geometric.data import Data, HeteroData

class Hypergraph:
    """
    Hypergraph class represents undirected hypepergraphs. It contains six
    important attributes:

    1. `hyperedges` -- torch.Tensor matrix with two rows, where first row
    represents node id and second row represents hyperedge id. It encodes
    belongingness of specific node to specific hyperedge.

    2. `labels` -- torch.Tensor vector which contains a category class (non-negative
    integer) for every node in the hypergraph. Can be None if no labels are available.
    
    3. `hyperedge_weight` -- torch.Tensor vector which contains a hyperedge weight
    for every hyperedge in the hypergraph. Can be None, if no hyperedge weights are
    available.

    The remaining three attributes are `train_mask`, `val_mask`, and `test_mask`
    which are binary vectors that have value 1 for nodes that should be used
    in training/validation/test process and 0 if they should not be used in
    the corresponding process.
    """
    def __init__(self, hyperedges: torch.Tensor, labels: torch.Tensor=None,
                 hyperedge_weight: torch.Tensor=None, train_mask: torch.Tensor=None,
                 val_mask: torch.Tensor=None, test_mask: torch.Tensor=None) -> None:
        """
        Initializes undirected hypergraph object, which describes undirected hypergraph.
        
        :param self:
        :param hyperedges: torch.Tensor matrix with two rows, where first row
        represents node id and second row represents hyperedge id. It encodes
        belongingness of specific node to specific hyperedge.
        :type hyperedges: torch.Tensor
        :param labels: torch.Tensor vector which contains a category class (non-negative
        integer) for every node in the hypergraph. Can be None if no labels are available.
        :type labels: torch.Tensor
        :param hyperedge_weight: torch.Tensor vector which contains a hyperedge weight
        for every hyperedge in the hypergraph. Can be None, if no hyperedge weights are
        available.
        :type hyperedge_weight: torch.Tensor
        :param train_mask: torch.Tensor binary vector which contains 1 for nodes that
        should be used in training and 0 otherwise.
        :type train_mask: torch.Tensor
        :param val_mask: torch.Tensor binary vector which contains 1 for nodes that
        should be used in validation process and 0 otherwise.
        :type val_mask: torch.Tensor
        :param test_mask: torch.Tensor binary vector which contains 1 for nodes that
        should be used in testing process and 0 otherwise.
        :type test_mask: torch.Tensor
        """
        super().__init__()
        # validating input for being tensor
        if not isinstance(hyperedges, torch.Tensor):
            raise ValueError(f"Expected 'hyperedges' variable to be of type torch.Tensor, received {hyperedges.__class__}")
        if not isinstance(labels, torch.Tensor) and labels!=None:
            raise ValueError(f"Expected 'labels' variable to be of type torch.Tensor, received {labels.__class__}")
        if not isinstance(hyperedge_weight, torch.Tensor) and hyperedge_weight!=None:
            raise ValueError(f"Expected 'hyperedge_weight' variable to be of type torch.Tensor, received {hyperedge_weight.__class__}")
        if not isinstance(train_mask, torch.Tensor) and train_mask!=None:
            raise ValueError(f"Expected 'train_mask' variable to be of type torch.Tensor, received {train_mask.__class__}")
        if not isinstance(val_mask, torch.Tensor) and val_mask!=None:
            raise ValueError(f"Expected 'val_mask' variable to be of type torch.Tensor, received {val_mask.__class__}")
        if not isinstance(test_mask, torch.Tensor) and test_mask!=None:
            raise ValueError(f"Expected 'test_mask' variable to be of type torch.Tensor, received {test_mask.__class__}")
        # validating input according to the format rules of the corresponding argument
        if len(hyperedges.shape)!=2 or hyperedges.shape[0]!=2 or torch.is_floating_point(hyperedges) or torch.is_complex(hyperedges) or hyperedges.dtype==torch.bool:
            raise ValueError(f"'hyperedges' argument is provided in invalid format. Expected integer matrix with 2 rows, received tensor of shape {hyperedges.shape} and data type {hyperedges.dtype}")
        if labels!=None and (torch.is_floating_point(labels) or torch.is_complex(labels)):
            raise ValueError("'labels' tensor argument cannot be of floating or complex data types.")
        if torch.any(hyperedges<0):
            raise ValueError("'hyperedges' cannot contain negative values.")
        if labels!=None and torch.any(labels<0):
            raise ValueError("Labels must be non-negative integers or booleans.")
        if train_mask!=None and (train_mask.dtype!=torch.bool or len(train_mask)!=1):
            raise ValueError(f"Train mask must be a boolean vector, obtained tensor with data type {train_mask.dtype} and shape {train_mask.shape}")
        if val_mask!=None and (val_mask.dtype!=torch.bool or len(val_mask)!=1):
            raise ValueError(f"Validation mask must be a boolean vector, obtained tensor with data type {val_mask.dtype} and shape {val_mask.shape}")
        if test_mask!=None and (test_mask.dtype!=torch.bool or len(test_mask)!=1):
            raise ValueError(f"Test mask must be a boolean vector, obtained tensor with data type {test_mask.dtype} and shape {test_mask.shape}")

        # setting the number of nodes and edges
        self.num_nodes = int(hyperedges[0].max().item()) + 1
        if labels!=None:
            temp_num_nodes = labels.shape[0]
            if temp_num_nodes>=self.num_nodes:
                self.num_nodes = temp_num_nodes
            else:
                raise ValueError(f"'hyperedges' argument define more nodes ({self.num_nodes}) than described by 'labels' ({temp_num_nodes}).")
        self.num_edges = int(hyperedges[1].max().item()) + 1
        if hyperedge_weight!=None:
            temp_num_edges = hyperedge_weight.shape[0]
            if temp_num_edges>=self.num_edges:
                self.num_edges = temp_num_edges
            else:
                raise ValueError(f"'hyperedges' argument define more hyperedges ({self.num_edges}) than described by 'hyperedge_weight' ({temp_num_edges}).")

        # validating labels and hyperedge_weight vectors
        if labels!=None and (len(labels.shape)!=1 or labels.shape[0]!=self.num_nodes):
            breakpoint()
            raise ValueError(f"'labels' must be a vector of size {self.num_nodes}, received tensor with shape {labels.shape}")
        if hyperedge_weight!=None and (len(hyperedge_weight.shape)!=1 or hyperedge_weight.shape[0]!=self.num_edges):
            raise ValueError(f"'hyperedge_weight' must be a vector of size {self.num_edges}, received tensor with shape {hyperedge_weight.shape}")
        if train_mask!=None and train_mask.shape[0]!=self.num_nodes:
            raise ValueError(f"Train mask vector must be of size {self.num_nodes}, received {train_mask.shape[0]}")
        if val_mask!=None and val_mask.shape[0]!=self.num_nodes:
            raise ValueError(f"Validation mask vector must be of size {self.num_nodes}, received {val_mask.shape[0]}")
        if test_mask!=None and test_mask.shape[0]!=self.num_nodes:
            raise ValueError(f"Test mask vector must be of size {self.num_nodes}, received {test_mask.shape[0]}")
        
        self.hyperedges = hyperedges
        self.labels = labels
        self.hyperedge_weight = hyperedge_weight
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
    
    def incidence_matrix(self) -> torch.Tensor:
        """
        Generates dense incidence matrix of the current hypergraph. For
        sparse incidence matrix refer to sparse_incidence_matrix.
        :return: Incidence matrix of the hypergraph.
        :rtype: Tensor
        """
        incidence_matrix = torch.zeros((self.num_nodes, self.num_edges), dtype=bool)
        for i in range(self.hyperedges.shape[1]):
            node = int(self.hyperedges[0, i].item())
            hyperedge = int(self.hyperedges[1, i].item())
            incidence_matrix[node, hyperedge] = 1 if self.hyperedge_weight==None else self.hyperedge_weight[hyperedge]
        return incidence_matrix
    
    def sparse_incidence_matrix(self) -> torch.sparse.SparseSemiStructuredTensor:
        """
        Sparse incidence matrix of the current hypergraph. Sparsity type is COO,
        as it is the most natural based on the internal hypergraph representation.
        :return: Sparse incidence matrix of the hypergraph.
        :rtype: SparseSemiStructuredTensor
        """
        if self.hyperedge_weight!=None:
            return torch.sparse_coo_tensor(self.hyperedges, self.hyperedge_weight[self.hyperedges[1]], (self.num_nodes, self.num_edges))
        else:
            return torch.sparse_coo_tensor(self.hyperedges, torch.ones((self.hyperedges.shape[1],)), (self.num_nodes, self.num_edges))
    
    def clique_graph(self) -> Data:
        """
        Generates a clique graph representation from the current hypergraph.
        :return: Clique graph representation of the hypergraph. Besides `edge_index`
        and `y` common for Data class, `weight_matrix` of size
        `(self.num_nodes, self.num_nodes)` defines weight of each edge in the graph,
        `train_mask`, `val_mask`, and `test_mask` define mask for nodes to be applied
        during the corresponding operations.
        :rtype: Data
        """
        sparse_inc = self.sparse_incidence_matrix().float() ** 0.5
        weight_matrix = torch.sparse.mm(sparse_inc, sparse_inc.T)
        weight_matrix.values()[weight_matrix.indices()[0]==weight_matrix.indices()[1]] = 0
        edge_index = weight_matrix.indices()
        return Data(edge_index=edge_index, weight_matrix=weight_matrix,
                    y=self.labels, train_mask=self.train_mask, val_mask=self.val_mask,
                    test_mask=self.test_mask)

    def incidence_graph(self):
        """
        Generates an incidence graph representation from the current hypergraph.
        :return: Incidence graph representation containing `self.num_nodes+self.num_edges`
        number of nodes. First `self.num_nodes` represent hypergraph nodes, the
        rest nodes represent hyperedges, which is reflected by the `is_hypergraph_node`
        vector. As it common for Data object, defines edge_index, which describes
        edges between nodes in the incidence graph. Also includes `weight_matrix`
        that defines a weight of each edge in the incidence graph. For hypergraph nodes,
        the object `hypergraph_nodes` includes labels in `y` and `train_mask`, `val_mask`,
        and `test_mask` masks.
        :rtype: Data
        """
        edge_index = self.hyperedges.clone()
        edge_index_weights = self.hyperedge_weight[edge_index[1]]
        edge_index[1] += self.num_nodes
        edge_index_swapped = torch.empty_like(edge_index)
        edge_index_swapped[0] = edge_index[1]
        edge_index_swapped[1] = edge_index[0]
        edge_index = torch.cat([edge_index, edge_index_swapped], 1)
        edge_index_weights = torch.cat([edge_index_weights, edge_index_weights], 0)
        weight_matrix = torch.sparse_coo_tensor(edge_index, edge_index_weights, (self.num_nodes+self.num_edges, self.num_nodes+self.num_edges))
        is_hypergraph_node = torch.zeros((self.num_nodes+self.num_edges,), dtype=bool)
        is_hypergraph_node[:self.num_nodes] = True
        return HeteroData(edge_index=edge_index, weight_matrix=weight_matrix,
                          is_hypergraph_node=is_hypergraph_node,
                          hypergraph_nodes = {
                              'y': self.labels, 'train_mask': self.train_mask,
                              'val_mask': self.val_mask, 'test_mask': self.test_mask
                          })