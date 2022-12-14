import numpy as np
import pytorch_lightning as pl
import torch
import tqdm
from segmentation_models_pytorch.encoders import get_encoder
from torch import nn
from models.misc import SizeNormLayer
from models.proposed_method.utils import OPS, DEFAULT_PRIMITIVES, SMALL_PRIMITIVES, CONV3_PRIMITIVES
from utils.helpers import compute_metrics

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


class Supernet(pl.LightningModule):
    # A supernet is a directed-acyclic graph that consists of nodes and edges.
    def __init__(self, encoder, num_class=5, layers=8,
                 loss_fn=nn.CrossEntropyLoss(), lr=0.0001, selection_epochs=5,
                 ops_set="default", condition_metric='val_acc'):
        super().__init__()
        self.save_hyperparameters()

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
        self.ch_each_scale = self.encoder.out_channels
        self.num_depth = len(self.ch_each_scale)
        self.num_layers = layers
        self.num_class = num_class
        self.loss_fn = loss_fn
        self.edge_ids = []
        self.node_ids = []
        self.finalized = False

        # Print some information about the search configs
        print(f"Number of layers: {self.num_layers}")
        print(f"Number of depths: {self.num_depth}")
        print(f"Number of classes: {self.num_class}")
        print(f"Number of channel at each scale: {self.ch_each_scale}")
        print(f"Number of candidate ops: {len(PRIMITIVES)}")
        print(f"Candidate ops: {PRIMITIVES}")

        # Contruct the nodes
        self.layers_nodes = nn.ModuleList()
        for l in range(self.num_layers):  # layers
            # Last layer don't have any up or down-sampling
            if (l + 1) == self.num_layers:
                self.layers_nodes.append(nn.ModuleList([Node(in_ch) for in_ch in self.ch_each_scale]))
                for s in range(self.num_depth):
                    self.edge_ids.append(f"{l},{s},same")
                    self.node_ids.append(f"{l},{s}")
            else:  # All other normal cases
                tmp = nn.ModuleList()
                for s, in_ch in enumerate(self.ch_each_scale):  # loop through each scale
                    # First scale don't have upsampling
                    if s == 0:
                        tmp.append(Node(in_ch, down_ch=self.ch_each_scale[s+1]))
                        self.edge_ids.append(f"{l},{s},same")
                        self.edge_ids.append(f"{l},{s},down")
                        self.node_ids.append(f"{l},{s}")
                    # Last scale don't have downsampling
                    elif s == (self.num_depth - 1):
                        tmp.append(Node(in_ch, up_ch=self.ch_each_scale[s-1]))
                        self.edge_ids.append(f"{l},{s},same")
                        self.edge_ids.append(f"{l},{s},up")
                        self.node_ids.append(f"{l},{s}")
                    else:
                        tmp.append(Node(in_ch, down_ch=self.ch_each_scale[s+1], up_ch=self.ch_each_scale[s-1]))
                        self.edge_ids.append(f"{l},{s},same")
                        self.edge_ids.append(f"{l},{s},up")
                        self.edge_ids.append(f"{l},{s},down")
                        self.node_ids.append(f"{l},{s}")
                self.layers_nodes.append(tmp)

        # Output
        self.output_layer = StaggeredOutput(self.ch_each_scale, self.num_class)

        # Helper layers
        self.norm_input = SizeNormLayer(mode='pad', divisible_number=2 ** self.num_depth)
        self.recovery_size = SizeNormLayer(mode='crop')

        # Set condition metric for arch selection
        if condition_metric == 'val_acc':
            self.monitor_metric = 'micro_acc'
        elif condition_metric == 'val_mIOU':
            self.monitor_metric = 'macro_iou'

    def on_save_checkpoint(self, checkpoint):
        if self.finalized is True:
            for layer_name in checkpoint['state_dict'].copy().keys():
                if 'selected_op' not in layer_name:
                    del checkpoint['state_dict'][layer_name]

            checkpoint['nas-candidate'] = True

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


    def configure_optimizers(self):
        lr = self.hparams.lr
        opt = torch.optim.Adam(self.parameters(), lr)
        return opt

    def find_best_arch(self, train_loader, val_loader, device='cuda', max_edges=2):
        '''
        :param train_loader:
        :param val_loader:
        :param device:
        :param max_edges: number of edges allowed per node
        :return:
        '''
        ########## Nested function #######
        # Train loop
        def train_val_loop(mode='train'):
            if mode == 'train':
                self.train()
                acc = 0
                for batch in tqdm.tqdm(train_loader, desc=mode):
                    raw, label = batch['raw'].to(device), batch['label'].to(device)
                    output = self(raw)

                    opt.zero_grad()
                    loss = self.loss_fn(output, label)
                    loss.backward()
                    opt.step()

                    # Calculate acc
                    pred_max = torch.argmax(output, dim=1)
                    metrics = compute_metrics(pred_max, label)
                    acc += metrics[self.monitor_metric].item()

                return acc / len(train_loader)
            elif mode == 'val':
                self.eval()
                acc = 0
                for batch in tqdm.tqdm(val_loader, desc=mode):
                    raw, label = batch['raw'].to(device), batch['label'].to(device)
                    with torch.no_grad():
                        output = self(raw)

                        # Calculate acc
                        pred_max = torch.argmax(output, dim=1)
                        metrics = compute_metrics(pred_max, label)
                        acc += metrics[self.monitor_metric].item()

                return acc / len(val_loader)
        ########## END Nested function #######

        # Parameters
        opt = self.configure_optimizers()
        epochs = self.hparams.selection_epochs

        # Perform arch discretization
        print("Starting operation selection.")
        while len(self.edge_ids) > 0:
            print(f"{len(self.edge_ids)} edges left!")
            rand_id = np.random.randint(len(self.edge_ids))
            edge_id = self.edge_ids.pop(rand_id)
            l, s, path = edge_id.split(',')
            target_node = self.layers_nodes[int(l)][int(s)]
            print(f"Edge id of {edge_id} selected.")

            if path == 'same':
                target_edge = target_node.edge
            elif path == 'up':
                target_edge = target_node.up_edge
            elif path == 'down':
                target_edge = target_node.down_edge

            # PT-based selection
            all_accs = []
            for op_id in range(len(PRIMITIVES)):
                target_edge.mask_op(op_id, selection_process=True)  # remove the target op
                acc = train_val_loop(mode='val')
                all_accs.append(acc)

            # Discretize the target_edge
            norm_criteria = self.normalize_criteria(all_accs, flipped=True)
            print("Accs:", norm_criteria)

            selected_op_id = np.argmax(norm_criteria)
            target_edge.discretize_edge(selected_op_id)
            print(f"Discretize edge {edge_id} to op {selected_op_id}")

            # Train for few epochs to stabilize Supernet after discretization
            print("Training the supernet after discretization.")
            for _ in range(epochs):
                train_val_loop(mode='train')

        # Topology selection
        if max_edges < 3:
            print("Starting topology selection.")
            while len(self.node_ids) > 0:
                print(f"{len(self.node_ids)} nodes left!")
                rand_id = np.random.randint(len(self.node_ids))
                node_id = self.node_ids.pop(rand_id)
                l, s = node_id.split(',')
                target_node = self.layers_nodes[int(l)][int(s)]
                print(f"Node id of {node_id} selected.")

                all_accs = []
                for edge_id in range(3):
                    if target_node.edge_exist(edge_id):
                        target_node.mask_edge(edge_id)  # remove the target edge
                        acc = train_val_loop(mode='val')
                        all_accs.append(acc)
                    else:
                        all_accs.append(1)  # put max acc for node that don't exists

                # Pruning edges
                norm_criteria = self.normalize_criteria(all_accs, flipped=True)
                print("Accs:", norm_criteria)
                selected_edge_ids = np.argsort(norm_criteria)[-max_edges:]  # take the last two elements  (the second largest and largest)
                target_node.discretize_node(selected_edge_ids)
                print(f"Selected edges: {selected_edge_ids}")

                # Train for few epochs to stabilize Supernet after discretization
                print("Training the supernet after discretization.")
                for _ in range(epochs):
                    train_val_loop(mode='train')
        self.finalized = True

    def normalize_criteria(self, x, flipped=False):
        ''' Perform minmax normalization of a given input x'''
        min = np.min(x)
        max = np.max(x)
        x = (x - min) / (max - min)
        if flipped: x = 1-x
        return x


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
        self.register_buffer('pt_mask', torch.ones(3))  # a node has at most three outputs

    def forward(self, x, from_above_x=None, from_below_x=None):
        if from_above_x is not None:
            x += torch.nn.functional.interpolate(from_above_x, size=x.shape[2:])

        if from_below_x is not None:
            x += torch.nn.functional.interpolate(from_below_x, size=x.shape[2:])

        if self.down_ch is not None:
            down_out = self.down_edge(x) * self.pt_mask[1]
        else:
            down_out = None

        if self.up_ch is not None:
            up_out = self.up_edge(x) * self.pt_mask[2]
        else:
            up_out = None

        same_out = self.edge(x) * self.pt_mask[0]

        return same_out, down_out, up_out

    def edge_exist(self, id):
        if id == 0:
            return True   # same path always exists
        elif id == 1:
            return self.down_ch is not None
        elif id == 2:
            return self.up_ch is not None

    def num_edges(self):
        '''
        :return: the number of outgoing edges
        '''
        edges = 0
        for i in range(3):
            if self.edge_exist(i):
                edges += 1

        return edges


    def mask_edge(self, id):
        self.pt_mask = torch.ones_like(self.pt_mask)
        self.pt_mask[id] = 0

    def discretize_node(self, ids: list):
        self.pt_mask = torch.ones_like(self.pt_mask)

        for edge_id in range(3):
            if edge_id not in ids:
                if edge_id == 0:
                    self.edge.discretize_edge(PRIMITIVES.index('zero'))
                elif edge_id == 1:
                    if self.down_ch is not None: self.down_edge.discretize_edge(PRIMITIVES.index('zero'))
                elif edge_id == 2:
                    if self.up_ch is not None: self.up_edge.discretize_edge(PRIMITIVES.index('zero'))

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
        self.in_selection_process = False
        self.register_buffer('selected_op', torch.tensor(-100, dtype=torch.int))  # just an initial placeholder

        for primitive in PRIMITIVES:
            op = OPS[primitive](in_ch, out_ch, stride)
            self.ops.append(op)

        self.register_buffer('pt_mask', torch.ones(len(self.ops)))

    def forward(self, x):
        out = 0
        if self.in_selection_process:
            for op, mask in zip(self.ops, self.pt_mask):
                if mask == 1:
                    out += op(x) * mask
        else:
            for op, mask in zip(self.ops, self.pt_mask):
                out += op(x) * mask

        return out

    def mask_op(self, id, selection_process=False):
        self.pt_mask = torch.ones_like(self.pt_mask)
        self.pt_mask[id] = 0
        self.in_selection_process = selection_process

    def discretize_edge(self, id):
        self.in_selection_process = False
        self.pt_mask = torch.zeros_like(self.pt_mask)
        self.pt_mask[id] = 1
        self.selected_op = torch.tensor(id, device=self.pt_mask.device, dtype=torch.int)

        # Remove all unused operatons for efficiency
        for i in range(len(self.ops)):
            if i != id:
                self.ops[i] = OPS['zero'](self.in_ch, self.out_ch, self.stride)
