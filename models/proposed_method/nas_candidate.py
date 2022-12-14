import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from segmentation_models_pytorch.encoders import get_encoder
from torch import nn
from models.misc import SizeNormLayer
from models.proposed_method.utils import OPS, EDGE_COLORS, \
    DEFAULT_PRIMITIVES, SMALL_PRIMITIVES, CONV3_PRIMITIVES
from utils.helpers import compute_metrics, show_visual_results
PRIMITIVES = None

class StaggeredOutput(nn.Module):
    def __init__(self, ch_each_scale, num_classes):
        super().__init__()
        self.ch_each_scale = ch_each_scale
        self.num_classes = num_classes

        self.convs = nn.ModuleList()
        for s, ch in enumerate(ch_each_scale):
            if s==0:
                self.convs.append(nn.Conv2d(ch, num_classes, kernel_size=3, padding='same'))
            else:
                self.convs.append(nn.Conv2d(ch, ch_each_scale[s-1], kernel_size=3, padding='same'))


    def forward(self, output_at_each_scale):
        # output = x
        prev_out = None
        for conv, out in zip(self.convs[::-1], output_at_each_scale[::-1]):
            out = out[0]  # output only as same_out path
            if prev_out != None:
                out = out + torch.nn.functional.interpolate(prev_out, size=out.shape[2:])
            prev_out = conv(out)

        return prev_out


class NASCandidate(pl.LightningModule):
    # A supernet is a directed-acyclic graph that consists of nodes and edges.
    def __init__(self, encoder, num_class=5, layers=8,
                 loss_fn=nn.CrossEntropyLoss(), lr=0.0001, ops_set="default"):
        super().__init__()
        self.save_hyperparameters()
        self.cleaned = False

        global PRIMITIVES
        if ops_set == 'default':
            PRIMITIVES = DEFAULT_PRIMITIVES
        elif ops_set == 'small':
            PRIMITIVES = SMALL_PRIMITIVES
        elif ops_set == 'conv3':
            PRIMITIVES = CONV3_PRIMITIVES
        else:
            raise NotImplementedError

        self.encoder = get_encoder(name=encoder, in_channels=25,
                                   depth=5, weights="imagenet")

        # Set hyperameters
        self.ch_each_scale = self.encoder.out_channels  # remove the input image ch
        self.num_depth = len(self.ch_each_scale)
        self.num_layers = layers
        self.num_class = num_class
        self.loss_fn = loss_fn
        self.edge_ids = []

        # Print some information about the search configs
        print(f"Number of layers: {self.num_layers}")
        print(f"Number of depths: {self.num_depth}")
        print(f"Number of classes: {self.num_class}")

        # Contruct the nodes
        self.layers_nodes = nn.ModuleList()
        for l in range(self.num_layers):  # layers
            # Last layer don't have any up or down-sampling
            if (l + 1) == self.num_layers:
                self.layers_nodes.append(nn.ModuleList([Node(in_ch) for in_ch in self.ch_each_scale]))
                for s in range(self.num_depth):
                    self.edge_ids.append(f"{l},{s},same")
            else:  # All other normal cases
                tmp = nn.ModuleList()
                for s, in_ch in enumerate(self.ch_each_scale):  # loop through each scale
                    # First scale don't have upsampling
                    if s == 0:
                        tmp.append(Node(in_ch, down_ch=self.ch_each_scale[s+1]))
                        self.edge_ids.append(f"{l},{s},same")
                        self.edge_ids.append(f"{l},{s},down")
                    # Last scale don't have downsampling
                    elif s == (self.num_depth - 1):
                        tmp.append(Node(in_ch, up_ch=self.ch_each_scale[s-1]))
                        self.edge_ids.append(f"{l},{s},same")
                        self.edge_ids.append(f"{l},{s},up")
                    else:
                        tmp.append(Node(in_ch, down_ch=self.ch_each_scale[s+1], up_ch=self.ch_each_scale[s-1]))
                        self.edge_ids.append(f"{l},{s},same")
                        self.edge_ids.append(f"{l},{s},up")
                        self.edge_ids.append(f"{l},{s},down")
                self.layers_nodes.append(tmp)

        # Output
        self.output_layer = StaggeredOutput(self.ch_each_scale, self.num_class)

        # Helper layers
        self.norm_input = SizeNormLayer(mode='pad', divisible_number=2 ** self.num_depth)
        self.recovery_size = SizeNormLayer(mode='crop')

    def on_load_checkpoint(self, checkpoint):
        if 'nas-candidate' not in checkpoint:
            raise AttributeError("This is not a candidate nas weight.")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['nas-candidate'] = True

    def plot_arch(self):
        # Loop the nodes
        fig = plt.figure()
        for l, nodes in enumerate(self.layers_nodes):
            for s, node in enumerate(nodes):
                # If this node has same path
                if PRIMITIVES[node.edge.selected_op] != 'zero':
                    x_axis = [l, l+1]
                    y_axis = [s, s]

                    edge_color = EDGE_COLORS[PRIMITIVES[node.edge.selected_op]]
                    if PRIMITIVES[node.edge.selected_op] == 'identity':
                        plt.plot(x_axis, y_axis, lw=2.5,
                                 color=edge_color,
                                 linestyle='--', marker='o', markersize=15,
                                 markeredgecolor='black',
                                 markerfacecolor='w')
                    else:
                        plt.plot(x_axis, y_axis, lw=2.5,
                                 color=edge_color,
                                 linestyle='-', marker='o', markersize=15,
                                 markeredgecolor='black',
                                 markerfacecolor='w')

                    if (l + 1) == self.num_layers:  # if last layer
                        plt.plot(l+1, s, marker='s', color='black', markersize=17)

                # If this node has up path
                if node.up_ch is not None:
                    if PRIMITIVES[node.up_edge.selected_op] != 'zero':
                        x_axis = [l, l+1]
                        y_axis = [s, s-1]

                        edge_color = EDGE_COLORS[PRIMITIVES[node.up_edge.selected_op]]

                        if PRIMITIVES[node.up_edge.selected_op] == 'identity':
                            plt.plot(x_axis, y_axis, lw=2.5,
                                     color=edge_color,
                                     linestyle='--', marker='o', markersize=15,
                                     markeredgecolor='black',
                                     markerfacecolor='w')
                        else:
                            plt.plot(x_axis, y_axis, lw=2.5,
                                     color=edge_color,
                                     linestyle='-', marker='o', markersize=15,
                                     markeredgecolor='black',
                                     markerfacecolor='w')

                # If this node has down path
                if node.down_ch is not None:
                    if PRIMITIVES[node.down_edge.selected_op] != 'zero':
                        x_axis = [l, l+1]
                        y_axis = [s, s+1]
                        x_axis.append(l + 1)
                        y_axis.append(s+1)

                        edge_color = EDGE_COLORS[PRIMITIVES[node.down_edge.selected_op]]
                        if PRIMITIVES[node.down_edge.selected_op] == 'identity':
                            plt.plot(x_axis, y_axis, lw=2.5,
                                     color=edge_color,
                                     linestyle='--', marker='o', markersize=15,
                                     markeredgecolor='black',
                                     markerfacecolor='w')
                        else:
                            plt.plot(x_axis, y_axis, lw=2.5,
                                     color=edge_color,
                                     linestyle='-', marker='o', markersize=15,
                                     markeredgecolor='black',
                                     markerfacecolor='w')
        # Plot styling
        plt.xlabel("Layers")
        plt.xticks(np.array(range(self.num_layers)), np.arange(1, self.num_layers+1))
        plt.ylabel("Scales")
        plt.yticks(np.array(range(self.num_depth)), np.array(range(self.num_depth)))
        plt.gca().invert_yaxis()
        return fig

    def cleanup_optimum_dnn(self, remove_dead_node=False):
        '''
        Remove unused ops after loading the optimum model arch.
        :return:
        '''
        if self.cleaned is False:
            if remove_dead_node == True:
                pruned = False
                while True:
                    # Backward flow
                    for l in reversed(range(self.num_layers)):
                        if l+1 == self.num_layers:  # last layer
                            continue

                        zero_op_index = PRIMITIVES.index('zero')
                        for s, node in enumerate(self.layers_nodes[l]):
                            # Same path
                            if self.layers_nodes[l+1][s].isActive() == False:
                                success = node.edge.discretize_edge(zero_op_index)
                                if success: pruned=True

                            # Down path
                            if node.down_ch is not None and self.layers_nodes[l+1][s+1].isActive() == False:
                                success = node.down_edge.discretize_edge(zero_op_index)
                                if success: pruned = True

                            # Up path
                            if node.up_ch is not None and self.layers_nodes[l+1][s-1].isActive() == False:
                                success = node.up_edge.discretize_edge(zero_op_index)
                                if success: pruned = True

                    # Forward flow
                    for l in range(self.num_layers):
                        if l == 0:  # first layer
                            continue

                        zero_op_index = PRIMITIVES.index('zero')
                        for s, node in enumerate(self.layers_nodes[l]):
                            # Check if this node has inputs
                            no_inputs = False

                            if s == 0: # first scale
                                if not self.layers_nodes[l-1][s].edge_isActive(0) and not self.layers_nodes[l-1][s+1].edge_isActive(2):
                                    no_inputs = True
                            elif s+1 == self.num_depth: # last scale
                                if not self.layers_nodes[l - 1][s].edge_isActive(0) and not self.layers_nodes[l - 1][s - 1].edge_isActive(1):
                                    no_inputs = True
                            else:
                                if not self.layers_nodes[l - 1][s].edge_isActive(0) and not self.layers_nodes[l - 1][s - 1].edge_isActive(1) and not self.layers_nodes[l-1][s+1].edge_isActive(2):
                                    no_inputs = True

                            # If no inputs, set all edges to Zero
                            if no_inputs:
                                success = node.edge.discretize_edge(zero_op_index)
                                if success: pruned = True
                                if node.down_ch is not None:
                                    success = node.down_edge.discretize_edge(zero_op_index)
                                    if success: pruned = True
                                if node.up_ch is not None:
                                    success = node.up_edge.discretize_edge(zero_op_index)
                                    if success: pruned = True

                    if pruned is False:
                        break
                    else:
                        pruned = False

            for l, nodes in enumerate(self.layers_nodes):
                for s, node in enumerate(nodes):
                    # Same path
                    node.edge.discretize_edge(node.edge.selected_op.item())

                    # Down path
                    if node.up_ch is not None:
                        node.up_edge.discretize_edge(node.up_edge.selected_op.item())

                    # Up path
                    if node.down_ch is not None:
                        node.down_edge.discretize_edge(node.down_edge.selected_op.item())

            self.cleaned = True

    def setup(self, stage):
        self.cleanup_optimum_dnn(remove_dead_node=True)

    def forward(self, x):
        x = self.norm_input(x)
        encoder_outputs = self.encoder(x)
        prev_results = [(encoder_outputs[s], None, None) for s in range(self.num_depth)]
        for l, nodes in enumerate(self.layers_nodes):
            # Results of current layer
            cur_results = [(None, None, None) for _ in range(self.num_depth)]

            # Loop scales - vertical axis
            if l == 0: # first layer
                for s, node in enumerate(nodes):
                    cur_results[s] = node(prev_results[s][0])
            else:  # other layers
                for s, node in enumerate(nodes):
                    if s == 0:  # first scale
                        cur_results[s] = node(prev_results[s][0], None,
                                                      prev_results[s + 1][2])
                    elif (s+1) == self.num_depth:  # last scale
                        cur_results[s] = node(prev_results[s][0],
                                              prev_results[s - 1][1], None)
                    else:  # other normal cases
                        cur_results[s] = node(prev_results[s][0],
                                              prev_results[s - 1][1],
                                              prev_results[s + 1][2])

            prev_results = cur_results.copy()  # Store the outputs for this layer.

        # Output layer
        output = self.output_layer(prev_results)
        output = self.recovery_size(output, self.norm_input.padding)

        return output

    def training_step(self, batch, batch_idx):
        raw, label = batch['raw'], batch['label']
        output = self(raw)

        loss = self.loss_fn(output, label)
        pred_max = torch.argmax(output, dim=1)
        metrics = compute_metrics(pred_max, label)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", metrics['micro_acc'], on_epoch=True)
        self.log("train_mAcc", metrics['macro_acc'], on_epoch=True)
        self.log("train_mIOU", metrics['macro_iou'], on_epoch=True)
        self.log("train_mDice", metrics['macro_dice'], on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        raw, label = batch['raw'], batch['label']
        output = self(raw)

        loss = self.loss_fn(output, label)
        pred_max = torch.argmax(output, dim=1)
        metrics = compute_metrics(pred_max, label)

        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", metrics['micro_acc'], on_epoch=True)
        self.log("val_mAcc", metrics['macro_acc'], on_epoch=True)
        self.log("val_mIOU", metrics['macro_iou'], on_epoch=True)
        self.log("val_mDice", metrics['macro_dice'], on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        raw, label = batch['raw'], batch['label']
        output = self(raw)

        loss = self.loss_fn(output, label)
        pred_max = torch.argmax(output, dim=1)
        metrics = compute_metrics(pred_max, label)

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", metrics['micro_acc'], on_epoch=True)
        self.log("test_mAcc", metrics['macro_acc'], on_epoch=True)
        self.log("test_mIOU", metrics['macro_iou'], on_epoch=True)
        self.log("test_mDice", metrics['macro_dice'], on_epoch=True)

        # Log figure
        fig = show_visual_results(raw.detach().cpu().numpy(),
                            label.detach().cpu().numpy(),
                            output.detach().cpu().numpy(),
                            show_visual=False)
        self.logger.experiment.log_figure(figure_name=f"Result {batch_idx}",
                                          figure=fig)
        return metrics

    def test_epoch_end(self, outputs):
        mIOUs = []
        mDices = []
        mAccs = []
        accs = []
        for output in outputs:
            mIOUs.append(output['macro_iou'].item())
            mDices.append(output['macro_dice'].item())
            mAccs.append(output['macro_acc'].item())
            accs.append(output['micro_acc'].item())

        self.logger.experiment.log_asset_data(str(mIOUs), "mIOUs.txt")
        self.logger.experiment.log_asset_data(str(mDices), "mDices.txt")
        self.logger.experiment.log_asset_data(str(mAccs), "mAccs.txt")
        self.logger.experiment.log_asset_data(str(accs), "accs.txt")

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt = torch.optim.Adam(self.parameters(), lr)
        return opt


class Node(nn.Module):
    # A Node is a latent representation (eg. feature map).
    # Each node has at most three edges.
    def __init__(self, in_ch, up_ch=None, down_ch=None):
        super().__init__()

        self.up_ch = up_ch
        self.down_ch = down_ch
        self.in_ch = in_ch

        if up_ch is not None:
            self.up_edge = Edge(in_ch, up_ch)

        if down_ch is not None:
            self.down_edge = Edge(in_ch, down_ch)

        self.edge = Edge(in_ch, in_ch)

    def forward(self, x, from_above_x=None, from_below_x=None):
        if from_above_x is not None:
            x += torch.nn.functional.interpolate(from_above_x, size=x.shape[2:])

        if from_below_x is not None:
            x += torch.nn.functional.interpolate(from_below_x, size=x.shape[2:])

        if self.down_ch is not None:
            down_out = self.down_edge(x)
        else:
            down_out = None

        if self.up_ch is not None:
            up_out = self.up_edge(x)
        else:
            up_out = None

        same_out = self.edge(x)

        return same_out, down_out, up_out

    def edge_isActive(self, edge_id):
        '''
        :param edge_id: 0: same, 1: down, 2: up
        :return:
        '''
        if edge_id == 0:
            return PRIMITIVES[self.edge.selected_op] != 'zero'
        elif edge_id == 1:
            return (self.down_ch is not None and PRIMITIVES[self.down_edge.selected_op] != 'zero')
        elif edge_id == 2:
            return (self.up_ch is not None and PRIMITIVES[self.up_edge.selected_op] != 'zero')

    def isActive(self):
        inactive_edge_count = 0

        # Same path
        if PRIMITIVES[self.edge.selected_op] == 'zero':
            inactive_edge_count += 1

        # Up path
        if self.up_ch is not None and PRIMITIVES[self.up_edge.selected_op] == 'zero':
            inactive_edge_count += 1
        elif self.up_ch is None:
            inactive_edge_count += 1

        # Down path
        if self.down_ch is not None and PRIMITIVES[self.down_edge.selected_op] == 'zero':
            inactive_edge_count += 1
        elif self.down_ch is None:
            inactive_edge_count += 1

        # Return the status
        if inactive_edge_count == 3:
            return False
        else:
            return True


class Edge(nn.Module):
    # Each edge is a mixed operation
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.ops = nn.ModuleList()
        self.register_buffer('selected_op', torch.tensor(-100, dtype=torch.int))  # just an initial placeholder

        for primitive in PRIMITIVES:
            op = OPS[primitive](in_ch, out_ch, stride)
            self.ops.append(op)

    def forward(self, x):
        out = 0

        for op in self.ops:
            out += op(x)

        return out

    def discretize_edge(self, id):
        if len(self.ops) > 1:  # if only one element, then this edge has been discretized already
            self.selected_op = torch.tensor(id, device=self.selected_op.device, dtype=torch.int)
            self.ops = nn.ModuleList([self.ops[id]])
            return True
        else:
            return False