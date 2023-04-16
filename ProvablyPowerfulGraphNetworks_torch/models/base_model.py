import torch
import torch.nn as nn
import layers.layers as layers
import layers.modules as modules
from torch.nn import BatchNorm1d as BN, BatchNorm2d as BN2


# def reset_child_params(module):
#     for layer in module.children():
#         if hasattr(layer, "reset_parameters"):
#             layer.reset_parameters()
#         reset_child_params(layer)


class BaseModel(nn.Module):
    def __init__(self, config):
        """
        Build the model computation graph, until scores/values are returned at the end
        """
        super().__init__()

        self.config = config
        use_new_suffix = config.architecture.new_suffix  # True or False
        block_features = (
            config.architecture.block_features
        )  # List of number of features in each regular block
        original_features_num = (
            config.node_labels + 1
        )  # Number of features of the input

        # First part - sequential mlp blocks
        last_layer_features = original_features_num
        self.reg_blocks = nn.ModuleList()
        for layer, next_layer_features in enumerate(block_features):
            mlp_block = modules.RegularBlock(
                config, last_layer_features, next_layer_features
            )
            self.reg_blocks.append(mlp_block)
            last_layer_features = next_layer_features

        # Second part
        self.fc_layers = nn.ModuleList()
        if use_new_suffix:
            for output_features in block_features:
                # each block's output will be pooled (thus have 2*output_features), and pass through a fully connected
                fc = modules.FullyConnected(
                    2 * output_features, self.config.num_classes, activation_fn=None
                )
                # fc = nn.Sequential(
                #     modules.FullyConnected(
                #         2 * output_features, self.config.num_classes, activation_fn=None
                #     ),
                #     BN(
                #         num_features=self.config.num_classes, momentum=0.9, affine=False
                #     ),
                # )
                self.fc_layers.append(fc)

        else:  # use old suffix
            # Sequential fc layers
            # self.fc_layers.append(modules.FullyConnected(2*block_features[-1], 512))
            # self.fc_layers.append(modules.FullyConnected(512, 256))
            # self.fc_layers.append(modules.FullyConnected(256, self.config.num_classes, activation_fn=None))

            self.fc_layers.append(modules.FullyConnected(2 * block_features[-1], 64))
            # self.fc_layers.append(BN(num_features=64, momentum=0.99, track_running_stats=False))
            self.fc_layers.append(modules.FullyConnected(64, 32))
            self.fc_layers.append(
                modules.FullyConnected(32, self.config.num_classes, activation_fn=None)
            )
        self.bn = BN(num_features=self.config.num_classes, momentum=1.0, affine=False)
        self.bn_layers = nn.ModuleList()
        for layer, next_layer_features in enumerate(block_features):
            self.bn_layers.append(BN2(num_features=next_layer_features, momentum=1.0, affine=False))

    def forward(self, input):
        x = input
        scores = torch.tensor(0, device=input.device, dtype=x.dtype)

        for i, block in enumerate(self.reg_blocks):

            x = block(x)
            x = self.bn_layers[i](x)

            if self.config.architecture.new_suffix:
                # use new suffix
                scores = self.fc_layers[i](layers.diag_offdiag_maxpool(x)) + scores

        if not self.config.architecture.new_suffix:
            # old suffix
            x = layers.diag_offdiag_maxpool(x)  # NxFxMxM -> Nx2F
            for fc in self.fc_layers:
                x = fc(x)
            scores = x
        scores = self.bn(scores)
        return scores

    # def reset_parameters(self):
    #     reset_child_params(self)
    #     return
