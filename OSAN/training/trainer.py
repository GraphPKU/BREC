import os
import pickle
from collections import defaultdict
from typing import Union, Optional, Any, List
from ml_collections import ConfigDict

import torch.linalg
from torch_geometric.data import Batch, Data

from data.custom_scheduler import BaseScheduler, StepScheduler
from data.data_utils import scale_grad, AttributedDataLoader, IsBetter
from data.metrics import eval_rocauc, eval_acc
from imle.noise import GumbelDistribution
from imle.target import TargetDistribution
from imle.wrapper import imle
from subgraph.construct import (edgemasked_graphs_from_nodemask,
                                edgemasked_graphs_from_undirected_edgemask,
                                edgemasked_graphs_from_directed_edgemask,
                                ordered_subgraph_construction,
                                construct_subgraph_batch, )
from training.imle_scheme import *

Optimizer = Union[torch.optim.Adam,
                  torch.optim.SGD]
Scheduler = Union[torch.optim.lr_scheduler.ReduceLROnPlateau,
                  torch.optim.lr_scheduler.MultiStepLR]
Emb_model = Any
Train_model = Any
Loss = Union[torch.nn.modules.loss.MSELoss, torch.nn.modules.loss.L1Loss]


class Trainer:
    def __init__(self,
                 dataset: str,
                 task_type: str,
                 voting: int,
                 max_patience: int,
                 criterion: Loss,
                 device: Union[str, torch.device],
                 imle_configs: ConfigDict,

                 sample_policy: str = 'node',
                 sample_k: int = -1,
                 remove_node: bool = True,
                 add_full_graph: bool = True,
                 **kwargs):
        super(Trainer, self).__init__()

        self.dataset = dataset
        self.task_type = task_type
        self.metric_comparator = IsBetter(self.task_type)
        self.voting = voting
        self.criterion = criterion
        self.device = device

        self.best_val_loss = 1e5
        self.best_val_metric = None
        self.patience = 0
        self.max_patience = max_patience

        self.curves = defaultdict(list)

        if imle_configs is not None:  # need to cache some configs, otherwise everything's in the dataloader already
            self.aux_loss_weight = imle_configs.aux_loss_weight
            self.imle_sample_rand = imle_configs.imle_sample_rand
            self.micro_batch_embd = imle_configs.micro_batch_embd
            self.imle_sample_policy = sample_policy
            self.remove_node = remove_node
            self.add_full_graph = add_full_graph
            self.temp = 1.
            self.target_distribution = TargetDistribution(alpha=1.0, beta=imle_configs.beta)
            self.noise_distribution = GumbelDistribution(0., imle_configs.noise_scale, self.device)
            self.noise_scale_scheduler = BaseScheduler(imle_configs.noise_scale)
            self.imle_scheduler = IMLEScheme(sample_policy,
                                             None,
                                             None,
                                             sample_k,
                                             return_list=False,
                                             perturb=False,
                                             sample_rand=imle_configs.imle_sample_rand)

    def clear_stats(self):
        self.curves = defaultdict(list)
        self.best_val_loss = 1e5
        self.best_val_metric = None
        self.patience = 0

    # def get_aux_loss(self, logits: torch.Tensor):
    #     """
    #     Aux loss that the sampled masks should be different
    #
    #     :param logits:
    #     :return:
    #     """
    #     logits = logits / torch.linalg.norm(logits, ord=None, dim=0, keepdim=True)
    #     eye = 1 - torch.eye(logits.shape[1], device=logits.device)
    #     loss = ((logits.t() @ logits) * eye).mean()
    #     return loss * self.aux_loss_weight

    def get_aux_loss(self, logits: torch.Tensor, split_idx: Tuple):
        """
        A KL divergence version
        """
        targets = torch.ones(logits.shape[0], device=logits.device, dtype=torch.float32)
        logits = torch.split(logits, split_idx, dim=0)
        targets = torch.split(targets, split_idx, dim=0)
        kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)
        loss = 0.
        for logit, target in zip(logits, targets):
            log_softmax_logits = torch.nn.LogSoftmax(dim=0)(logit.sum(1))
            target = target / logit.shape[0]
            loss += kl_loss(log_softmax_logits, target)
        return loss * self.aux_loss_weight

    def emb_model_forward(self, data: Union[Data, Batch], emb_model: Emb_model, train: bool) \
            -> Tuple[Union[Data, Batch], Optional[torch.FloatType]]:
        """
        Common forward propagation for train and val, only called when embedding model is trained.

        :param data:
        :param emb_model:
        :param train:
        :return:
        """
        logits_n, logits_e = emb_model(data)

        if self.imle_sample_policy in ['node',
                                       'node_ordered',
                                       'node_heuristic',
                                       'khop_subgraph',
                                       'khop_global',
                                       'khop_global_dual',
                                       'greedy_exp',
                                       'or',
                                       'or_optim']:
            split_idx = get_split_idx(data.ptr)
            logits = logits_n
            subgraphs_from_mask = edgemasked_graphs_from_nodemask
        elif self.imle_sample_policy in ['edge', 'mst']:
            split_idx = get_split_idx(data._slice_dict['edge_index'])
            logits = logits_e
            subgraphs_from_mask = edgemasked_graphs_from_undirected_edgemask
        elif self.imle_sample_policy == 'edge_linegraph':
            split_idx = tuple(data.lin_num_nodes.cpu().tolist())
            logits = logits_e
            subgraphs_from_mask = edgemasked_graphs_from_directed_edgemask
        else:
            raise NotImplementedError

        graphs = Batch.to_data_list(data)

        self.imle_scheduler.graphs = graphs
        self.imle_scheduler.ptr = split_idx

        aux_loss = None
        if train:
            @imle(target_distribution=self.target_distribution,
                  noise_distribution=self.noise_distribution,
                  input_noise_temperature=self.temp,
                  target_noise_temperature=self.temp,
                  nb_samples=1)
            def imle_sample_scheme(logits: torch.Tensor):
                return self.imle_scheduler.torch_sample_scheme(logits)

            sample_idx, aux_output = imle_sample_scheme(logits)
            if self.aux_loss_weight > 0:
                aux_loss = self.get_aux_loss(sample_idx, split_idx)
            self.noise_distribution.scale = self.noise_scale_scheduler()
        else:
            sample_idx, aux_output = self.imle_scheduler.torch_sample_scheme(logits)

        if aux_output is None:
            # unordered
            list_subgraphs, edge_weights, selected_node_masks = subgraphs_from_mask(graphs=graphs,
                                                                                    edge_index=data.edge_index,
                                                                                    masks=sample_idx,
                                                                                    grad=train,
                                                                                    remove_node=self.remove_node,
                                                                                    add_full_graph=self.add_full_graph)
            data = construct_subgraph_batch(list_subgraphs,
                                            len(graphs),
                                            sample_idx.shape[-1] if not self.add_full_graph else sample_idx.shape[
                                                                                                     -1] + 1,
                                            edge_weights,
                                            selected_node_masks,
                                            self.device)
        else:
            # ordered
            data = ordered_subgraph_construction(self.dataset,
                                                 graphs,
                                                 sample_idx,
                                                 aux_output,
                                                 self.add_full_graph,
                                                 self.remove_node,
                                                 train)

        return data, aux_loss

    def train(self,
              dataloader: AttributedDataLoader,
              emb_model: Emb_model,
              model: Train_model,
              optimizer_embd: Optional[Optimizer],
              optimizer: Optimizer):

        if emb_model is not None:
            emb_model.train()
            self.imle_scheduler.return_list = False
            self.imle_scheduler.perturb = False
            self.imle_scheduler.sample_rand = self.imle_sample_rand
            optimizer_embd.zero_grad()

        model.train()
        train_losses = torch.tensor(0., device=self.device)
        if self.task_type != 'regression':
            preds = []
            labels = []
        else:
            preds, labels = None, None
        num_graphs = 0

        for batch_id, data in enumerate(dataloader.loader):
            data = data.to(self.device)
            print(data)
            optimizer.zero_grad()

            aux_loss = None
            if emb_model is not None:
                data, aux_loss = self.emb_model_forward(data, emb_model, train=True)

            pred = model(data)
            is_labeled = data.y == data.y
            loss = self.criterion(pred[is_labeled], data.y[is_labeled].to(torch.float))
            train_losses += loss.detach() * data.num_graphs  # aux loss not taken into account
            if aux_loss is not None:
                loss += aux_loss

            loss.backward()
            optimizer.step()
            if optimizer_embd is not None:
                if (batch_id % self.micro_batch_embd == self.micro_batch_embd - 1) or (batch_id >= len(dataloader) - 1):
                    emb_model = scale_grad(emb_model, (batch_id % self.micro_batch_embd) + 1)
                    optimizer_embd.step()
                    optimizer_embd.zero_grad()

            num_graphs += data.num_graphs
            if isinstance(preds, list):
                preds.append(pred)
                labels.append(data.y)

        train_loss = train_losses.item() / num_graphs
        self.curves['train_loss'].append(train_loss)

        if isinstance(preds, list):
            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)
            if self.task_type == 'rocauc':
                train_metric = eval_rocauc(labels, preds)
            elif self.task_type == 'acc':
                if preds.shape[1] == 1:
                    preds = (preds > 0.).to(torch.int)
                else:
                    preds = torch.argmax(preds, dim=1)
                train_metric = eval_acc(labels, preds)
            else:
                raise NotImplementedError
        else:
            train_metric = train_loss
        self.curves['train_metric'].append(train_metric)

        if emb_model is not None:
            del self.imle_scheduler.graphs
            del self.imle_scheduler.ptr

        return train_loss, train_metric

    @torch.no_grad()
    def inference(self,
                  dataloader: AttributedDataLoader,
                  emb_model: Emb_model,
                  model: Train_model,
                  scheduler_embd: Optional[Scheduler] = None,
                  scheduler: Optional[Scheduler] = None,
                  test: bool = False):
        if emb_model is not None:
            emb_model.eval()
            self.imle_scheduler.return_list = False
            self.imle_scheduler.perturb = self.voting > 1  # only perturb when voting more than once
            self.imle_scheduler.sample_rand = False  # test time, always take topk, inspite of noise perturbation

        model.eval()
        preds = []
        labels = []

        for v in range(self.voting):
            for data in dataloader.loader:
                data = data.to(self.device)

                if emb_model is not None:
                    data, _ = self.emb_model_forward(data, emb_model, train=False)

                pred = model(data)
                if dataloader.std is not None:
                    preds.append(pred * dataloader.std)
                    labels.append(data.y.to(torch.float) * dataloader.std)
                else:
                    preds.append(pred)
                    labels.append(data.y.to(torch.float))

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        is_labeled = labels == labels
        val_loss = self.criterion(preds[is_labeled], labels[is_labeled]).item()
        if self.task_type == 'rocauc':
            val_metric = eval_rocauc(labels, preds)
        elif self.task_type == 'regression':
            val_metric = val_loss
        elif self.task_type == 'acc':
            if preds.shape[1] == 1:
                preds = (preds > 0.).to(torch.int)
            else:
                preds = torch.argmax(preds, dim=1)
            val_metric = eval_acc(labels, preds)
        else:
            raise NotImplementedError

        early_stop = False
        if not test:
            self.curves['val_metric'].append(val_metric)
            self.curves['val_loss'].append(val_loss)

            self.best_val_loss = min(self.best_val_loss, val_loss)

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                raise NotImplementedError("Need to specify max or min plateau")
            else:
                scheduler.step()
            if scheduler_embd is not None:
                if isinstance(scheduler_embd, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    raise NotImplementedError("Need to specify max or min plateau")
                else:
                    scheduler_embd.step()

            if self.metric_comparator(val_metric, self.best_val_metric):
                self.best_val_metric = val_metric
                self.patience = 0
            else:
                self.patience += 1
                if self.patience > self.max_patience:
                    early_stop = True

        if emb_model is not None:
            del self.imle_scheduler.graphs
            del self.imle_scheduler.ptr

        return val_loss, val_metric, early_stop

    def save_curve(self, path):
        pickle.dump(self.curves, open(os.path.join(path, 'curves.pkl'), "wb"))
