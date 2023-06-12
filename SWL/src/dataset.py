from . import *


class GraphCount(data.InMemoryDataset):
    task_index = dict(
        triangle=0,
        tri_tail=1,
        star=2,
        cycle4=3,
        cycle5=4,
        cycle6=5,
    )

    def __init__(self, root: str, split: str, task: str, **kwargs):
        super().__init__(root=root, **kwargs)

        _pt = dict(zip(["train", "val", "test"], self.processed_paths))
        self.data, self.slices = torch.load(_pt[split])

        index = self.task_index[task]
        self.data.y = self.data.y[:, index : index + 1]

    @property
    def raw_file_names(self):
        return ["randomgraph.pt"]

    @property
    def processed_dir(self):
        return f"{self.root}/randomgraph"

    @property
    def processed_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    def process(self):
        (_pt,) = self.raw_file_names
        raw = torch.load(f"{self.root}/{_pt}")

        def to(graph):
            A = graph["A"]
            y = graph["y"]

            return pyg.data.Data(
                x=torch.ones(A.shape[0], 1, dtype=torch.int64),
                y=y,
                edge_index=torch.Tensor(np.vstack(np.where(graph["A"] > 0))).type(
                    torch.int64
                ),
            )

        data = [to(graph) for graph in raw["data"]]

        if self.pre_filter is not None:
            data = filter(self.pre_filter, data)

        if self.pre_transform is not None:
            data = map(self.pre_transform, data)

        data_list = list(data)
        normalize = torch.std(torch.stack([data.y for data in data_list]), dim=0)

        for split in ["train", "val", "test"]:
            from operator import itemgetter

            split_idx = raw["index"][split]
            splits = itemgetter(*split_idx)(data_list)

            data, slices = self.collate(splits)
            data.y = data.y / normalize

            torch.save((data, slices), f"{self.processed_dir}/{split}.pt")
