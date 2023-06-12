from .esan_model import GNN, DSnetwork
from .GINE_gnn import NetGINE, NetGINE_ordered
from .GCN_embd import GCN_emb
from .GCN_edge_embd import GCN_edge_emb
from .GIN_embd import GINE_embd
from .GINE_alchemy import NetGINEAlchemy
from .ogb_mol_gnn import OGBGNN, OGBGNN_order
from .GINE_qm9 import NetGINE_QM, NetGINE_QM_ordered
from .pna import PNANet, PNANet_order
from data.const import DATASET_FEATURE_STAT_DICT, MAX_NUM_NODE_DICT


def get_model(args, train_set):
    if args.model.lower() == 'gine':
        model = NetGINE(DATASET_FEATURE_STAT_DICT[args.dataset]['node'],
                        DATASET_FEATURE_STAT_DICT[args.dataset]['edge'],
                        args.hid_size,
                        args.dropout,
                        args.num_convlayers,
                        jk=args.gnn_jk,
                        num_class=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'])
    elif args.model.lower() == 'gine_ordered_forward':
        model = NetGINE_ordered(DATASET_FEATURE_STAT_DICT[args.dataset]['node'],
                                DATASET_FEATURE_STAT_DICT[args.dataset]['edge'],
                                args.hid_size,
                                MAX_NUM_NODE_DICT[args.dataset],
                                args.extra_feature_hidden,
                                args.dropout,
                                args.num_convlayers,
                                jk=args.gnn_jk,
                                num_class=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'])
    elif args.model.lower() == 'ogb_gin_ordered':
        model = OGBGNN_order(gnn_type='gin',
                             num_tasks=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
                             num_layer=args.num_convlayers,
                             emb_dim=args.hid_size,
                             extra_in_dim=MAX_NUM_NODE_DICT[args.dataset],
                             extra_encode_dim=args.extra_feature_hidden,
                             drop_ratio=args.dropout,
                             virtual_node=False,
                             graph_pooling='sum')
    elif args.model.lower() == 'gine_alchemy':
        model = NetGINEAlchemy(DATASET_FEATURE_STAT_DICT[args.dataset]['node'],
                               DATASET_FEATURE_STAT_DICT[args.dataset]['edge'],
                               args.hid_size,
                               num_class=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
                               num_layers=args.num_convlayers)
    elif args.model.lower() == 'gin-virtual':
        model = OGBGNN(gnn_type='gin',
                       num_tasks=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
                       num_layer=args.num_convlayers,
                       emb_dim=args.hid_size,
                       drop_ratio=args.dropout,
                       virtual_node=True)
    elif args.model.lower() == 'ogb_gin':
        model = OGBGNN(gnn_type='gin',
                       num_tasks=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
                       num_layer=args.num_convlayers,
                       emb_dim=args.hid_size,
                       drop_ratio=args.dropout,
                       virtual_node=False)
    elif args.model.lower() == 'ogb_originalgin':
        model = OGBGNN(gnn_type='originalgin',
                       num_tasks=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
                       num_layer=args.num_convlayers,
                       emb_dim=args.hid_size,
                       drop_ratio=args.dropout,
                       virtual_node=False)
    elif args.model.lower() == 'gine_qm9':
        model = NetGINE_QM(DATASET_FEATURE_STAT_DICT[args.dataset]['node'],
                           DATASET_FEATURE_STAT_DICT[args.dataset]['edge'],
                           args.hid_size,
                           args.num_convlayers,
                           DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'])
    elif args.model.lower() == 'gine_qm9_ordered':
        model = NetGINE_QM_ordered(DATASET_FEATURE_STAT_DICT[args.dataset]['node'],
                                   DATASET_FEATURE_STAT_DICT[args.dataset]['edge'],
                                   args.hid_size,
                                   args.num_convlayers,
                                   MAX_NUM_NODE_DICT[args.dataset.lower()],
                                   args.extra_feature_hidden,
                                   DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'])
    elif args.model.lower() == 'dsnet':  # ESAN's model
        subgraph_gnn = GNN(gnn_type=args.gnn_type, num_tasks=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
                           num_layer=args.num_convlayers, in_dim=DATASET_FEATURE_STAT_DICT[args.dataset]['node'],
                           emb_dim=args.hid_size, drop_ratio=args.dropout, JK=args.jk,
                           graph_pooling='sum' if args.gnn_type != 'gin' else 'mean',
                           )
        model = DSnetwork(subgraph_gnn=subgraph_gnn,
                          channels=args.channels,
                          num_tasks=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
                          invariant=args.dataset.lower() == 'zinc')
    elif args.model.lower() == 'pna':
        model = PNANet(train_set,
                       DATASET_FEATURE_STAT_DICT[args.dataset]['node'],
                       DATASET_FEATURE_STAT_DICT[args.dataset]['edge'],
                       args.hid_size,
                       args.edge_hid_size,
                       args.num_convlayers,
                       DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'])
    elif args.model.lower() == 'pna_ordered':
        model = PNANet_order(train_set,
                             DATASET_FEATURE_STAT_DICT[args.dataset]['node'],
                             MAX_NUM_NODE_DICT[args.dataset],
                             args.extra_feature_hidden,
                             DATASET_FEATURE_STAT_DICT[args.dataset]['edge'],
                             args.hid_size,
                             args.edge_hid_size,
                             args.num_convlayers,
                             DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'])
    else:
        raise NotImplementedError

    if args.imle_configs is not None:
        if 'embd_model' in args.imle_configs:
            if args.imle_configs['embd_model'] == 'gin':
                emb_model = GINE_embd(DATASET_FEATURE_STAT_DICT[args.dataset]['node'],
                                      DATASET_FEATURE_STAT_DICT[args.dataset]['edge'],
                                      args.hid_size,
                                      num_class=args.sample_configs.num_subgraphs, )
            else:
                raise NotImplementedError
        elif args.sample_configs.sample_policy == 'edge_linegraph':
            emb_model = GCN_edge_emb(DATASET_FEATURE_STAT_DICT[args.dataset]['node'],
                                     DATASET_FEATURE_STAT_DICT[args.dataset]['edge'],
                                     args.hid_size,
                                     args.sample_configs.num_subgraphs,
                                     normalize=args.imle_configs.norm_logits,
                                     encoder='ogb' in args.dataset.lower() or 'exp' in args.dataset.lower())
        else:
            emb_model = GCN_emb(DATASET_FEATURE_STAT_DICT[args.dataset]['node'],
                                DATASET_FEATURE_STAT_DICT[args.dataset]['edge'],
                                args.hid_size,
                                args.sample_configs.num_subgraphs,
                                normalize=args.imle_configs.norm_logits,
                                encoder='ogb' in args.dataset.lower() or 'exp' in args.dataset.lower() or 'brec' in args.dataset.lower())
    else:
        emb_model = None
        print('none here')

    return model, emb_model
