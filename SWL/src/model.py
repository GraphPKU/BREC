from .utils import *

# ---------------------------------------------------------------------------- #
#                                     MODEL                                    #
# ---------------------------------------------------------------------------- #


class NormReLU(nn.Sequential):
    def __init__(self, dim: int):
        super().__init__()

        self.add_module("bn", nn.BatchNorm1d(dim))
        self.add_module("ac", nn.ReLU())


class MLP(nn.Sequential):
    def __init__(self, idim: int, odim: int, hdim: int = None, norm: bool = True):
        super().__init__()
        hdim = hdim or idim

        self.add_module("input", nn.Linear(idim, hdim))
        self.add_module("input_nr", NormReLU(hdim) if norm else nn.ReLU())
        self.add_module("output", nn.Linear(hdim, odim))


# --------------------------------- EMBEDDING -------------------------------- #

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class Atom(nn.Module):
    """Atom encoder

    Args:
        dim (int): embedding dimension
        dis (int): maximum encoding distance
        encode (bool): whether to use encoder

    """

    def __init__(self, dim: int, max_dis: int, encode: bool = True):
        super().__init__()
        self.max_dis = max_dis
        self.encode = encode

        self.embed_v = encode and AtomEncoder(dim)
        if encode:
            print(
                "atom embedding parameters:",
                [param.numel() for param in self.embed_v.parameters()],
            )
        self.embed_d = nn.Embedding(max_dis + 1, dim)

    def forward(self, batch):
        if self.encode:
            x = self.embed_v(batch.x[:, None])
        else:
            x = 0
        d = self.embed_d(torch.clamp(batch.d, None, self.max_dis))

        # batch.x = torch.cat([x, d], dim=-1)
        batch.x = x + d
        del batch.d
        return batch


class Bond(nn.Module):
    """Bond encoder

    Args:
        dim (int): embedding dimension

    """

    def __init__(self, dim: int):
        super().__init__()

        self.embed = BondEncoder(dim)
        # print(
        #     "bond embedding parameters:",
        #     [param.numel() for param in self.embed.parameters()],
        # )

    def forward(self, message, attrs):
        if attrs is None:
            return F.relu(message)
        return F.relu(message + self.embed(attrs[:, None]))


# --------------------------------- AGGREGATE -------------------------------- #


def Agg(scheme: List[str], gin: bool = True):
    """Aggregation layer factory

    Args:
        scheme ([str]): aggregation scheme operators
        gin (bool, optional): if `True`, use GIN base encoder

    """

    class _(nn.Module):
        def __init__(self, idim: int, odim: int):
            super().__init__()

            self.eps = nn.ParameterDict(
                {agg: nn.Parameter(torch.zeros(1)) for agg in scheme}
            )

            self.enc = nn.ModuleDict({agg: Bond(idim) for agg in scheme if "L" in agg})

            self.mlp = nn.ModuleDict({agg: MLP(idim, odim) for agg in scheme})

            self.nr = NormReLU(odim)

        def aggregate(self, agg, batch):
            return self.mlp[agg](
                (batch.x * (1.0 + self.eps[agg]) if gin else 0.0)
                + aggregate(batch, agg, self.enc[agg] if agg in self.enc else None)
            )

        def forward(self, batch):
            batch.x = self.nr(sum(self.aggregate(agg, batch) for agg in scheme))
            return batch

    return _


# ---------------------------------- POOLING --------------------------------- #


class Pool(nn.Module):
    """Final pooling

    Args:
        idim (int): input dimension
        odim (int): output dimension

    """

    def __init__(self, idim: int, odim: int):
        super().__init__()

        self.predict = MLP(idim, odim, hdim=idim * 2, norm=False)

    def forward(self, batch):
        return self.predict(gnn.global_mean_pool(batch.x, batch.batch))


class GNN(nn.Sequential):
    """Subgraph-based GNN

    Args:
        idim (int): embedding dimension
        dis (int): maximum encoding distance
        As ([(nn.Module, int)]): aggregation schemes and hidden dimensions
        odim (int): final output dimension

    """

    def __init__(
        self,
        idim: int,
        odim: int,
        max_dis: int,
        encode: bool,
        As: List[Tuple[nn.Module, int]],
    ):
        super().__init__()

        self.add_module("encode", Atom(idim, max_dis, encode))

        for i, (A, dim) in enumerate(As):
            self.add_module(f"A{i}", A(idim, dim))
            idim = dim

        self.add_module("pool", Pool(idim, odim))
