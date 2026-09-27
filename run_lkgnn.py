from helper import *
from data_loader import *
from new_pyHGT_rel.model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from warnings import filterwarnings
filterwarnings('ignore')
import argparse
import numpy as np
import joblib
import gc
from collections import defaultdict
from ordered_set import OrderedSet
import copy
import os
import shelve
import torch.distributed as dist
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass
import dill, pickle

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get('BIONCL_DATA_ROOT', os.path.join(PROJECT_ROOT, 'data'))

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class Runner(object):

    def load_data(self):
        ent_set, rel_set = (OrderedSet(), OrderedSet())
        for split in ['train', 'test', 'valid']:
            for line in open(os.path.join(DATA_ROOT, self.p.dataset, f'{split}.txt')):
                sub, rel, obj = line.strip().split('\t')
                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)
        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.rel2id.update({rel + '_reverse': idx + len(self.rel2id) for idx, rel in enumerate(rel_set)})
        self.rel2id['self'] = len(self.rel2id)
        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}
        self.p.id2ent = self.id2ent
        self.p.id2rel = self.id2rel
        self.p.num_ent = len(self.ent2id)
        self.p.num_rel = len(self.rel2id)
        self.p.half_num_rel = (len(self.rel2id) - 1) // 2
        self.p.embed_dim = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim
        self.data = ddict(list)
        sr2o = ddict(set)
        train_sr2o = ddict(set)
        for split in ['train', 'test', 'valid']:
            for line in open(os.path.join(DATA_ROOT, self.p.dataset, f'{split}.txt')):
                sub, rel, obj = line.strip().split('\t')
                sub, rel, obj = (self.ent2id[sub], self.rel2id[rel], self.ent2id[obj])
                self.data[split].append((sub, rel, obj))
                if split == 'train':
                    train_sr2o[sub, rel].add(obj)
                    train_sr2o[obj, rel + self.p.half_num_rel].add(sub)
        self.data = dict(self.data)
        self.train_sr2o = {k: list(v) for k, v in train_sr2o.items()}
        for split in ['train', 'test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[sub, rel].add(obj)
                sr2o[obj, rel + self.p.half_num_rel].add(sub)
        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples = ddict(list)
        for split in ['train']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.half_num_rel
                self.triples['{}'.format(split)].append({'triple': (sub, rel, obj), 'label': self.train_sr2o[sub, rel], 'sub_samp': 1})
                self.triples['{}'.format(split)].append({'triple': (obj, rel_inv, sub), 'label': self.train_sr2o[obj, rel_inv], 'sub_samp': 1})
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.half_num_rel
                self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 'label': self.sr2o_all[sub, rel]})
                self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[obj, rel_inv]})
        self.triples = dict(self.triples)

        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            data = list(dataset_class(self.triples[split], self.p))
            if split == 'train':
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(data, shuffle=True, num_replicas=world_size, rank=local_rank)
                data_loader = DataLoader(data, batch_size, shuffle=False, num_workers=2, sampler=self.train_sampler, pin_memory=True, prefetch_factor=2)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(data, shuffle=False, num_replicas=world_size, rank=local_rank)
                data_loader = DataLoader(data, batch_size, shuffle=False, num_workers=2, sampler=sampler, pin_memory=True, prefetch_factor=2)
            return data_loader
        if local_rank == 0:
            self.data_iter = {'train': get_data_loader(TrainDataset, 'train', self.p.batch_size), 'valid_head': get_data_loader(TestDataset, 'valid_head', self.p.batch_size), 'valid_tail': get_data_loader(TestDataset, 'valid_tail', self.p.batch_size), 'test_head': get_data_loader(TestDataset, 'test_head', self.p.batch_size), 'test_tail': get_data_loader(TestDataset, 'test_tail', self.p.batch_size)}
        else:
            self.data_iter = {'train': get_data_loader(TrainDataset, 'train', self.p.batch_size), 'valid_head': get_data_loader(TestDataset, 'valid_head', self.p.batch_size), 'valid_tail': get_data_loader(TestDataset, 'valid_tail', self.p.batch_size)}
        self.edge_index, self.edge_type, self.rel_attribute, self.node_attribute, self.type_matrix, self.adj = self.construct_adj()
        if self.p.pretrained_emb:
            self.entity_embeddings, self.relation_embeddings = self.init_embeddings(os.path.join(self.p.dataset, 'entity2vec.txt'), os.path.join(self.p.dataset, 'relation2vec.txt'))
            print('Initialised relations and entities from TransE')
        else:
            self.entity_embeddings = np.random.randn(self.p.num_ent, 200)
            self.relation_embeddings = np.random.randn(self.p.num_rel, 200)
            print('Initialised relations and entities randomly')

    def init_embeddings(self, entity_file, relation_file):
        entity_emb, relation_emb = ([], [])
        with open(os.path.join(DATA_ROOT, entity_file)) as f:
            for line in f:
                entity_emb.append([float(val) for val in line.strip().split()])
        with open(os.path.join(DATA_ROOT, relation_file)) as f:
            for line in f:
                relation_emb.append([float(val) for val in line.strip().split()])
        return (np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32))

    def construct_adj(self):
        edge_index, edge_type = ([], [])
        for sub, rel, obj in self.data['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)
        for sub, rel, obj in self.data['train']:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.half_num_rel)
        for en in self.id2ent:
            edge_index.append((en, en))
            edge_type.append(self.p.num_rel - 1)
        edge_index = torch.LongTensor(edge_index).to(self.device).t()
        edge_type = torch.LongTensor(edge_type).to(self.device)
        type2id, id2type, node2type = ({}, {}, {})
        type_matrix = []
        node_attribute = []
        rel_attribute = []
        ent2newid = dict(sorted(self.ent2id.items(), reverse=False, key=lambda kv: kv[1]))
        rel2newid = dict(sorted(self.rel2id.items(), reverse=False, key=lambda kv: kv[1]))
        if self.p.use_nodetype:
            with open(os.path.join(DATA_ROOT, self.p.dataset, 'maintype2id.txt')) as f:
                for i in f:
                    tg = i.rstrip('\n').split('\t')
                    type2id[tg[0]] = int(tg[1])
                    id2type[int(tg[1])] = tg[0]
            self.type2id = type2id
            self.id2type = id2type
            self.p.num_nodetype = len(self.type2id)
            with open(os.path.join(DATA_ROOT, self.p.dataset, 'node2maintype.txt')) as f:
                for i in f:
                    tg = i.rstrip('\n').split('\t')
                    node2type[tg[0]] = tg[1]
            self.node2type = node2type
            for ent in ent2newid:
                nodetype_id = type2id[node2type[ent]]
                type_matrix.append(nodetype_id)
        else:
            self.p.num_nodetype = 1
            type_matrix = [0] * len(self.ent2id)
        type_matrix = torch.LongTensor(type_matrix).view(-1, 1).to(self.device)
        if self.p.use_node_pretrain:
            with shelve.open(os.path.join(DATA_ROOT, self.p.dataset, 'TextEmbedding', 'bert-large-uncased512dim', self.p.style + '_bert-large-uncased512dim')) as f:
                for i in ent2newid.keys():
                    node_attribute.append(f[i])
            node_attribute = torch.FloatTensor(node_attribute).to(self.device)
        else:
            node_attribute = None
        if self.p.use_relation_pretrain:
            with shelve.open('path_to_relation_embeddings') as f:
                for i in rel2newid.keys():
                    rel_attribute.append(f[i])
            rel_attribute = torch.FloatTensor(rel_attribute).to(self.device)
        else:
            rel_attribute = None
        data = []
        rows = []
        columns = []
        rows = rows + [i for i in range(self.p.num_ent)]
        columns = columns + [i for i in range(self.p.num_ent)]
        data = data + [self.p.num_rel - 1 for i in range(self.p.num_ent)]
        indices = torch.LongTensor([rows, columns])
        v = torch.LongTensor(data)
        adj = [indices, v, self.p.num_ent]
        return (edge_index, edge_type, rel_attribute, node_attribute, type_matrix, adj)

    def __init__(self, params):
        self.p = params
        self.logger = get_logger(args.task_name + '_' + args.name + '.log', args.log_dir, args.config_dir)
        self.logger.setLevel(logging.DEBUG)
        for arg in vars(args):
            self.logger.info((arg, getattr(args, arg)))
        self.logger.info('\n\n')
        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')
        self.load_data()
        self.model = self.add_model(self.p.score_func)
        if self.p.lr_scheduler == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.p.max_epochs)
        elif self.p.lr_scheduler == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min')
        elif self.p.lr_scheduler == 'CyclicLR':
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.01, max_lr=0.1)
        elif self.p.lr_scheduler == 'OneCycleLR':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.01, steps_per_epoch=len(self.data_iter['train']), epochs=self.p.max_epochs)
        elif self.p.lr_scheduler == 'CosineAnnealingWarmRestarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=1)

    def add_model(self, score_func):
        self.gnn = GNN(self.edge_index, self.edge_type, self.rel_attribute, self.node_attribute, self.type_matrix, self.adj, params=self.p).cuda(local_rank)
        self.matcher = Matcher(self.p, self.p.n_hid).cuda(local_rank)
        self.model = nn.Sequential(self.gnn, self.matcher).cuda(local_rank)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], find_unused_parameters=True, output_device=local_rank, broadcast_buffers=False)
        try:
            self.model = torch.compile(self.model, mode='max-autotune', dynamic=True, fullgraph=True, backend='inductor')
        except Exception as e:
            print('not use pytorch2.0 compile, skip')
            pass
        self.model = self.model.module
        if self.p.save_path != None:
            print('retrain')
            self.optimizer = self.add_optimizer(self.model.parameters())
            self.load_model(os.path.join(self.p.model_dir, self.p.save_path))
            num_param = sum([p.numel() for p in self.model.parameters()])
            self.logger.info(f'Number of parameters: {num_param / 1000000.0:.4f} M')
        else:
            print('train')
            self.optimizer = self.add_optimizer(self.model.parameters())
            num_param = sum([p.numel() for p in self.model.parameters()])
            self.logger.info(f'Number of parameters: {num_param / 1000000.0:.4f} M')
        return self.model

    def add_optimizer(self, parameters):
        if self.p.lr_scheduler != 'CyclicLR':
            if args.optimizer == 'adamw':
                optimizer = torch.optim.AdamW(parameters, weight_decay=self.p.l2)
            elif args.optimizer == 'adam':
                optimizer = torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)
            elif args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(parameters, lr=0.001, weight_decay=self.p.l2)
            elif args.optimizer == 'adagrad':
                optimizer = torch.optim.Adagrad(parameters, weight_decay=self.p.l2)
        else:
            optimizer = torch.optim.SGD(parameters, lr=0.001, weight_decay=self.p.l2)
        return optimizer

    def read_batch(self, batch, split):
        if split == 'train':
            triple, label = [_.to(self.device) for _ in batch]
            return (triple[:, 0], triple[:, 1], triple[:, 2], label)
        else:
            triple, label = [_.to(self.device) for _ in batch]
            return (triple[:, 0], triple[:, 1], triple[:, 2], label)

    def save_model(self, save_path):
        state = {'state_dict': self.model.state_dict(), 'best_val': self.best_val, 'best_epoch': self.best_epoch, 'optimizer': self.optimizer.state_dict(), 'args': vars(self.p)}
        torch.save(state, save_path)

    def load_model(self, load_path):
        state = torch.load(load_path, map_location=f'cuda:{local_rank}')
        state_dict = state['state_dict']
        self.best_val = state['best_val']
        self.best_val_mrr = self.best_val['mrr']
        self.model.load_state_dict(state_dict)
        self.optimizer.param_groups[0]['lr'] = state['optimizer']['param_groups'][0]['lr']

    def evaluate(self, split):
        if split == 'valid':
            left_results, left_loss = self.predict(split=split, mode='tail_batch')
            right_results, right_loss = self.predict(split=split, mode='head_batch')
            valid_loss = (left_loss + right_loss) / 2
            results = get_combined_results(left_results, right_results)
            return (results, valid_loss)
        else:
            left_results = self.predict(split=split, mode='tail_batch')
            right_results = self.predict(split=split, mode='head_batch')
            results = get_combined_results(left_results, right_results)
            return results

    def predict(self, split='valid', mode='tail_batch'):
        self.model.eval()
        with torch.no_grad():
            results = {}
            losses = []
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])
            for step, batch in enumerate(train_iter):
                if split == 'valid':
                    sub, rel, obj, label = self.read_batch(batch, 'valid')
                    x, y, r, _, _ = self.gnn.forward(sub, rel)
                    pred = self.matcher.forward(x, r, y)
                    if self.p.loss_function == 'contrastive_loss':
                        loss = self.contrastive_loss(pred, label, self.p.tau_plus, self.p.beta, self.p.estimator, obj)
                    elif self.p.loss_function == 'mask_softmax':
                        loss = self.mask_softmax(pred, label, obj)
                    elif self.p.loss_function == 'mix_loss':
                        loss = self.mix_loss(pred, label, self.p.tau_plus, self.p.beta, self.p.estimator, obj)
                    elif self.p.loss_function == 'bcel_loss':
                        loss = self.bcel_loss(pred, label)
                    losses.append(loss.item())
                else:
                    sub, rel, obj, label = self.read_batch(batch, split)
                    x, y, r, _, _ = self.gnn.forward(sub, rel)
                    pred = self.matcher.forward(x, r, y)
                b_range = torch.arange(pred.size()[0], device=self.device)
                target_pred = pred[b_range, obj]
                pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
                ranks = ranks.float()
                results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
                results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
                for k in range(10):
                    results['hits@{}'.format(k + 1)] = torch.numel(ranks[ranks <= k + 1]) + results.get('hits@{}'.format(k + 1), 0.0)
            if split == 'valid':
                valid_loss = np.mean(losses)
                return (results, valid_loss)
            else:
                return results

    def run_epoch(self, epoch, val_mrr=0):
        self.model.train()
        losses = []
        train_iter = iter(self.data_iter['train'])
        iters = len(self.data_iter['train'])
        self.train_sampler.set_epoch(int(epoch))
        for step, batch in enumerate(tqdm(train_iter, total=len(train_iter))):
            self.optimizer.zero_grad()
            sub, rel, obj, label = self.read_batch(batch, 'train')
            x, r, y = self.gnn.forward(sub, rel)
            pred = self.matcher.forward(x, r, y)
            if self.p.loss_function == 'contrastive_loss':
                loss = self.contrastive_loss(pred, label, self.p.tau_plus, self.p.beta, self.p.estimator, obj)
            elif self.p.loss_function == 'mask_softmax':
                loss = self.mask_softmax(pred, label, obj)
            elif self.p.loss_function == 'mix_loss':
                loss = self.mix_loss(pred, label, self.p.tau_plus, self.p.beta, self.p.estimator, obj)
            elif self.p.loss_function == 'bcel_loss':
                loss = self.bcel_loss(pred, label)
            if args.change_lr:
                torch.cuda.empty_cache()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()
                if self.p.lr_scheduler in ['CyclicLR', 'OneCycleLR']:
                    self.scheduler.step()
                elif self.p.lr_scheduler == 'CosineAnnealingWarmRestarts':
                    self.scheduler.step(epoch - 1 + step / iters)
            else:
                loss.backward()
                self.optimizer.step()
            losses.append(loss.item())
        loss = np.mean(losses)
        return loss

    def contrastive_loss_old(self, pred, label, tau_plus, beta, estimator):
        loss = 0
        pos = []
        for i in range(len(label)):
            index_line = torch.nonzero(label[i]).view(-1)
            rs = pred[i][index_line].sum()
            pos.append(rs.view(-1))
        pos = torch.stack(pos, dim=0)
        pos = torch.exp(pos / self.p.temperature)
        pos_neg = torch.exp(pred / self.p.temperature)
        if estimator == 'hard':
            N = LL - 1
            imp = (beta * neg.log()).exp()
            reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
            Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
            Ng = torch.clamp(Ng, min=N * np.e ** (-1 / args.temperature))
        elif estimator == 'easy':
            All = pos_neg.sum(dim=-1)
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')
        loss = (-torch.log(pos / All)).mean()
        return loss

    def isinf_nan_if(self, x):
        if torch.isinf(x).any():
            return 'inf'
        elif torch.isnan(x).any():
            return 'nan'
        else:
            return 'success'

    def isinf_nan(self, x):
        if torch.isinf(x).any():
            raise Exception(f'is inf:{x}')
        if torch.isnan(x).any():
            raise Exception(f'is nan:{x}')

    def contrastive_loss(self, pred, label, tau_plus, beta, estimator, obj):
        st = time.time()
        try:
            if self.isinf_nan_if(torch.exp(pred / self.p.temperature)) == 'inf' or self.isinf_nan_if(torch.exp(pred / self.p.temperature)) == 'nan':
                pred = pred / pred.max()
            if (torch.exp(pred / self.p.temperature) == 0).any():
                pred = torch.exp(pred / self.p.temperature) + 1e-08
            else:
                pred = torch.exp(pred / self.p.temperature)
            self.isinf_nan(pred)
            loss = 0
            pos = []
            for i in range(len(label)):
                if args.lbl_smooth != 0:
                    index_line = label[i] > 0.9
                else:
                    index_line = torch.nonzero(label[i]).view(-1)
                if self.p.sample_mod == 'full':
                    pos_random_idx = index_line
                elif self.p.sample_mod == 'part':
                    pos_random_idx = obj[i]
                else:
                    exit('please gain args.sample_mod')
                rs = pred[i][pos_random_idx].sum()
                pos.append(rs)
            pos = torch.stack(pos, dim=0)
            if estimator == 'hard':
                N = LL - 1
                imp = (beta * neg.log()).exp()
                reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
                Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
                Ng = torch.clamp(Ng, min=N * np.e ** (-1 / args.temperature))
            elif estimator == 'easy':
                neg = []
                index_line_bool = label <= 0
                for i in range(len(label)):
                    if args.lbl_smooth != 0:
                        index_line = label[i] < 0.1
                    else:
                        index_line = torch.nonzero(index_line_bool[i]).view(-1)
                    if self.p.sample_mod == 'full':
                        neg_random_idx = index_line
                    elif self.p.sample_mod == 'part':
                        neg_random_idx = index_line
                    else:
                        exit('please gain args.sample_mod')
                    rs = pred[i][neg_random_idx].sum()
                    neg.append(rs)
                neg = torch.stack(neg, dim=0)
            else:
                raise Exception('Invalid estimator selected. Please use any of [hard, easy]')
            if self.isinf_nan_if((-torch.log(pos / (pos + neg))).mean()) == 'inf' or self.isinf_nan_if((-torch.log(pos / (pos + neg))).mean()) == 'nan':
                pos = pos + 1e-08
                neg = neg + 1e-08
            loss = (-torch.log(pos / (pos + args.Q * neg))).mean()
            self.isinf_nan(loss)
        except Exception as e:
            print(e)
            exit('error')
        en = time.time()
        return loss

    def contrastive_loss_all(self, pred, label, tau_plus, beta, estimator, obj):
        st = time.time()
        try:
            if self.isinf_nan_if(torch.exp(pred / self.p.temperature)) == 'inf' or self.isinf_nan_if(torch.exp(pred / self.p.temperature)) == 'nan':
                pred = pred / pred.max()
            if (torch.exp(pred / self.p.temperature) == 0).any():
                pred = torch.exp(pred / self.p.temperature) + 1e-08
            else:
                pred = torch.exp(pred / self.p.temperature)
            self.isinf_nan(pred)
            loss = 0
            pos = []
            for i in range(len(label)):
                if args.lbl_smooth != 0:
                    index_line = label[i] > 0.9
                else:
                    index_line = torch.nonzero(label[i]).view(-1)
                if self.p.sample_mod == 'full':
                    pos_random_idx = index_line
                elif self.p.sample_mod == 'part':
                    pos_random_idx = obj[i]
                else:
                    exit('please gain args.sample_mod')
                rs = pred[i][pos_random_idx].sum()
                pos.append(rs)
            pos = torch.stack(pos, dim=0)
            if estimator == 'hard':
                N = LL - 1
                imp = (beta * neg.log()).exp()
                reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
                Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
                Ng = torch.clamp(Ng, min=N * np.e ** (-1 / args.temperature))
            elif estimator == 'easy':
                neg = []
                index_line_bool = label <= 0
                for i in range(len(label)):
                    if args.lbl_smooth != 0:
                        index_line = label[i] < 0.1
                    else:
                        index_line = torch.nonzero(index_line_bool[i]).view(-1)
                    if self.p.sample_mod == 'full':
                        neg_random_idx = index_line
                    elif self.p.sample_mod == 'part':
                        neg_random_idx = index_line
                    else:
                        exit('please gain args.sample_mod')
                    rs = pred[i][neg_random_idx].sum()
                    neg.append(rs)
                neg = torch.stack(neg, dim=0)
            else:
                raise Exception('Invalid estimator selected. Please use any of [hard, easy]')
            if self.isinf_nan_if((-torch.log(pos / (pos + neg))).mean()) == 'inf' or self.isinf_nan_if((-torch.log(pos / (pos + neg))).mean()) == 'nan':
                pos = pos + 1e-08
                neg = neg + 1e-08
            pos = pos.sum()
            neg = neg.sum()
            loss = -torch.log(pos / (pos + args.Q * neg)) / label.shape[0]
            self.isinf_nan(loss)
        except Exception as e:
            print(e)
            exit('error')
        en = time.time()
        return loss

    def norm_data(self, score):
        score = score / self.p.embed_dim
        score_norm = torch.norm(score, p=2, dim=-1, keepdim=True)
        score = score / score_norm
        return score

    def mask_softmax_old(self, pred, label):
        loss = 0
        pos = []
        pred_log = torch.log_softmax(pred, dim=-1)
        for i in range(len(label)):
            index_line = torch.nonzero(label[i]).view(-1)
            rs = pred[i][index_line].sum()
            pos.append(rs.view(-1))
        loss = torch.cat(pos, dim=0).mean()
        return -loss

    def mask_softmax(self, pred, label, obj):
        loss = 0
        pos = []
        pred_log = torch.log_softmax(pred, dim=-1) / self.p.embed_dim
        for i in range(len(label)):
            index_line = obj[i]
            rs = pred_log[i][index_line].sum()
            pos.append(rs)
        loss = torch.stack(pos, dim=0).mean()
        return -loss

    def bcel_loss(self, pred, label):
        bceloss = torch.nn.BCELoss()
        pred_s = torch.sigmoid(pred)
        return bceloss(pred_s, label)

    def mix_loss(self, pred, label, tau_plus, beta, estimator, obj):
        if args.mix_loss_mod == 'CLoss-MLoss':
            all_loss = self.contrastive_loss(pred, label, tau_plus, beta, estimator, obj) + self.mask_softmax(pred, label, obj)
        elif args.mix_loss_mod == 'CLoss-BCELoss':
            all_loss = self.contrastive_loss(pred, label, tau_plus, beta, estimator, obj) + self.bcel_loss(pred, label)
        elif args.mix_loss_mod == 'MLoss-BCELoss':
            all_loss = self.mask_softmax(pred, label, obj) + self.bcel_loss(pred, label)
        return all_loss

    def test_old(self):
        self.logger.info('Loading best model, Evaluating on Test data')
        save_path = os.path.join(args.model_dir, args.save_path)
        self.load_model(save_path)
        test_results = self.evaluate('test')
        self.logger.info('\n\n')
        self.logger.info('Test:')
        self.logger.info('Test_mrr: %.5f' % test_results['mrr'])
        self.logger.info('Test_mr: %.5f' % test_results['mr'])
        self.logger.info('Test_hit1: %.5f' % test_results['hits@1'])
        self.logger.info('Test_hit3: %.5f' % test_results['hits@3'])
        self.logger.info('Test_hit10: %.5f' % test_results['hits@10'])
        localtime = time.asctime(time.localtime(time.time()))
        self.logger.info(f'Program end time: {localtime}')

    def test(self):
        print('Testing')
        self.logger.info('Loading best model, Evaluating on Test data')
        save_path = os.path.join(args.model_dir, args.save_path)
        self.load_model(save_path)
        if args.store_emb == 1:
            flag = None
            st = 0
            ed = 100000
            all_data = False
            analy_results = self.analy(flag, st, ed, all_data)
            print('Embeddings are saved separately because of their size')
            name_list = ['ent_emb_dict', 'one_candidate_entity_score', 'true_candidate_entity_score', 'predict_candidate_entity', 'all_candidate_entity_score'][st:ed]
            for dt, name in zip(analy_results, name_list):
                base_file = f'{args.store_name}_{args.dataset}_{name}_analy'
                self.save_dict_in_100mb_chunks(dt, target_mb=1000, prefix=base_file)
                del dt
                gc.collect()
        else:
            test_results = self.evaluate('test')
            self.logger.info('\n\n')
            self.logger.info('test:')
            self.logger.info('Test_mrr: %.5f' % test_results['mrr'])
            self.logger.info('Test_mr: %.5f' % test_results['mr'])
            self.logger.info('Test_hit1: %.5f' % test_results['hits@1'])
            self.logger.info('Test_hit3: %.5f' % test_results['hits@3'])
            self.logger.info('Test_hit10: %.5f' % test_results['hits@10'])
        localtime = time.asctime(time.localtime(time.time()))
        self.logger.info(f'stop time:{localtime}')

    def analy(self, split, st=0, ed=1000, all_data=True):
        results = self.store_data(split='test', mode='head_batch', st=st, ed=ed, all_data=all_data)
        right_results = self.store_data(split='test', mode='tail_batch', st=st, ed=ed, all_data=all_data)
        for i, j in zip(results, right_results):
            i.update(j)
        return results

    def store_data(self, split='valid', mode='tail_batch', st=0, ed=1000, all_data=True):
        ent_emb_dict = {}
        all_candidate_entity_score = {}
        true_candidate_entity_score = {}
        one_candidate_entity_score = {}
        predict_candidate_entity = {}
        self.model.eval()
        with torch.no_grad():
            results = {}
            losses = []
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])
            for step, batch in enumerate(tqdm(train_iter)):
                sub, rel, obj, label = self.read_batch(batch, split)
                sub_emb, all_ent, all_rel, init_all_ent, init_all_rel = self.gnn.forward(sub, rel)
                pred = self.matcher.forward(sub_emb, all_rel, all_ent)
                multi_pred = pred.clone()
                b_range = torch.arange(pred.size()[0], device=self.device)
                target_pred = pred[b_range, obj]
                pred = torch.where(label.bool(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)
                ranks = ranks[b_range, obj]
                ranks = ranks.float()
                rr = 1.0 / ranks
                for ids, (i, j, k, rr_score) in enumerate(zip(sub.detach().cpu().numpy(), rel.detach().cpu().numpy(), obj.detach().cpu().numpy(), rr.detach().cpu().numpy())):
                    one_candidate_entity_score[self.id2ent[i], self.id2rel[j], self.id2ent[k]] = rr_score
                if not ent_emb_dict:
                    for ids, (i, j) in enumerate(zip(all_ent.detach().cpu().numpy(), init_all_ent.detach().cpu().numpy())):
                        ent_name = self.id2ent[ids]
                        ent_emb_dict[ent_name] = [j, i]
                for ids, (i, j, k, score, lab) in enumerate(zip(sub.detach().cpu().numpy(), rel.detach().cpu().numpy(), obj.detach().cpu().numpy(), multi_pred.detach().cpu().numpy(), label.bool())):
                    all_tmp_dict = {}
                    true_tmp_dict = {}
                    key = (self.id2ent[i], self.id2rel[j], self.id2ent[k])
                    value_score = score
                    value_name = list(self.id2ent.values())
                    for k, v, lb in zip(value_name, value_score, lab):
                        all_tmp_dict[k] = v
                        if lb:
                            true_tmp_dict[k] = v
                    sorted_all_tmp_dict = dict(sorted(all_tmp_dict.items(), key=lambda item: item[1], reverse=True))
                    sorted_true_tmp_dict = dict(sorted(true_tmp_dict.items(), key=lambda item: item[1], reverse=True))
                    all_candidate_entity_score[key] = sorted_all_tmp_dict
                    true_candidate_entity_score[key] = sorted_true_tmp_dict
                    pred_name = list(sorted_all_tmp_dict.keys())[0]
                    true_name_list = set(list(sorted_true_tmp_dict.keys()))
                    if pred_name in true_name_list:
                        flag = True
                    else:
                        flag = False
                    predict_candidate_entity[key] = [pred_name, flag]
        return [ent_emb_dict, one_candidate_entity_score, true_candidate_entity_score, predict_candidate_entity, all_candidate_entity_score][st:ed]

    def fit(self):
        best_val = 0
        best_epoch = 0
        best_lr = 0
        best_mrr = 0
        best_mr = 0
        best_hit1 = 0
        best_hit3 = 0
        best_hit10 = 0
        val_mrr = 0
        self.best_val_mrr, self.best_val, self.best_epoch = (0.0, {}, 0)
        save_path = os.path.join(args.model_dir, args.task_name + '_' + args.name)
        stop_flag = False
        kill_cnt = 0
        for epoch in np.arange(self.p.max_epochs) + 1:
            starting = time.time()
            train_loss = self.run_epoch(epoch, val_mrr)
            if args.change_lr and self.p.lr_scheduler not in ['CyclicLR', 'OneCycleLR', 'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau']:
                self.scheduler.step()
            val_results, valid_loss = self.evaluate('valid')
            if local_rank == 0:
                ending = time.time()
                self.logger.info('Validation:')
                self.logger.info('Epoch:{}  time {:.3f}m  LR:{:.10f}'.format(epoch, (ending - starting) / 60, self.optimizer.param_groups[0]['lr']))
                self.logger.info('Train Loss:{:.10f}  Valid Loss:{:.10f}'.format(train_loss, valid_loss))
                self.logger.info('Valid MRR:{:.5f}  Valid MR:{:.5f}'.format(val_results['mrr'], val_results['mr']))
                self.logger.info('Valid H@1:{:.5f}  Valid H@3:{:.5f}  Valid H@10:{:.5f}'.format(val_results['hits@1'], val_results['hits@3'], val_results['hits@10']))
                self.logger.info('\n\n')
            if val_results['mrr'] > best_val:
                self.best_val = val_results
                self.best_val_mrr = val_results['mrr']
                self.best_epoch = epoch
                best_val = val_results['mrr']
                best_epoch = epoch
                best_mrr = val_results['mrr']
                best_mr = val_results['mr']
                best_hit1 = val_results['hits@1']
                best_hit3 = val_results['hits@3']
                best_hit10 = val_results['hits@10']
                best_lr = self.optimizer.param_groups[0]['lr']
                self.save_model(save_path)
                kill_cnt = 0
            else:
                kill_cnt += 1
                if kill_cnt % 10 == 0 and self.p.gamma > 5:
                    self.p.gamma -= 5
                    self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
                    self.logger.info('\n\n')
                if self.p.dynamic_temp:
                    if kill_cnt % 5 == 0 and self.p.save_path != None:
                        if self.p.temperature >= 1.0:
                            self.p.temperature = 0.2
                        else:
                            self.p.temperature += 0.1
                        self.logger.info('temperature add on saturation, updated value of temperature {}'.format(self.p.temperature))
                        self.logger.info('\n\n')
                    if kill_cnt % 5 == 0 and self.p.save_path == None:
                        if round(self.p.temperature) < 0.2:
                            self.p.temperature = 1.0
                        else:
                            self.p.temperature -= 0.1
                        self.logger.info('temperature decay on saturation, updated value of temperature {}'.format(self.p.temperature))
                        self.logger.info('\n\n')
                if local_rank == 0:
                    if kill_cnt > args.stop_num:
                        self.logger.info('Early Stopping!!')
                        stop_flag = True
                        break
            if local_rank == 0:
                if epoch % 5 == 0:
                    self.logger.info(f'best_epoch: {best_epoch}')
                    self.logger.info('best_lr: %.10f' % best_lr)
                    self.logger.info('best_mrr: %.5f' % best_mrr)
                    self.logger.info('best_mr: %.5f' % best_mr)
                    self.logger.info('best_hit1: %.5f' % best_hit1)
                    self.logger.info('best_hit3: %.5f' % best_hit3)
                    self.logger.info('best_hit10: %.5f' % best_hit10)
                    self.logger.info('\n\n')
        if local_rank == 0:
            self.logger.info(f'best_epoch: {best_epoch}')
            self.logger.info('best_lr: %.10f' % best_lr)
            self.logger.info('best_mrr: %.5f' % best_mrr)
            self.logger.info('best_mr: %.5f' % best_mr)
            self.logger.info('best_hit1: %.5f' % best_hit1)
            self.logger.info('best_hit3: %.5f' % best_hit3)
            self.logger.info('best_hit10: %.5f' % best_hit10)
            self.logger.info('\n\n')
            self.logger.info('Loading best model, Evaluating on Test data')
            torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.name))
            test_results = self.evaluate('test')
            self.logger.info('\n\n')
            self.logger.info('Test:')
            self.logger.info('Test_mrr: %.5f' % test_results['mrr'])
            self.logger.info('Test_mr: %.5f' % test_results['mr'])
            self.logger.info('Test_hit1: %.5f' % test_results['hits@1'])
            self.logger.info('Test_hit3: %.5f' % test_results['hits@3'])
            self.logger.info('Test_hit10: %.5f' % test_results['hits@10'])
            localtime = time.asctime(time.localtime(time.time()))
            self.logger.info(f'stop time:{localtime}')
            dist.broadcast(torch.tensor([0], dtype=torch.int).cuda(), src=rank)

    def data_generator(self, dt):
        for i in dt:
            yield dt[i]

    def get_approx_chunk_size(self, dictionary, target_mb=100):
        sample_item = next(iter(dictionary.items()))
        item_size = sys.getsizeof(sample_item[0]) + sys.getsizeof(sample_item[1])
        bytes_per_chunk = target_mb * 1024 * 1024
        return max(1, int(bytes_per_chunk / item_size))

    def save_dict_in_100mb_chunks(self, dictionary, target_mb=100, prefix='data_chunk'):
        chunk_size = self.get_approx_chunk_size(dictionary, target_mb)
        items = list(dictionary.items())
        for i in tqdm(range(0, len(items), chunk_size), desc='Saving 100MB chunks'):
            chunk = dict(items[i:i + chunk_size])
            chunk_file = f'{prefix}_{i // chunk_size}.pkl'
            with open(chunk_file, 'wb') as f:
                dill.dump(chunk, f, protocol=dill.HIGHEST_PROTOCOL)

    def load_all_chunks(self, prefix):
        full_dict = {}
        for chunk in self.stream_all_chunks(prefix):
            full_dict.update(chunk)
        return full_dict

    def stream_all_chunks(self, prefix):
        i = 0
        while True:
            chunk_file = f'{prefix}_{i}.pkl'
            if not os.path.exists(chunk_file):
                break
            yield self.load_single_chunk(chunk_file)
            i += 1

    def load_single_chunk(self, chunk_file: str):
        with open(chunk_file, 'rb') as f:
            return pickle.load(f)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training GNN For Link Predict (LP) Task')
    parser.add_argument('--name', default='', help='Set run name for saving/restoring models')
    parser.add_argument('--dataset', dest='dataset', default='FB15k-237', help='Dataset to use, default: FB15k-237')
    parser.add_argument('--model', dest='model', default='rhcgat', help='Model Name')
    parser.add_argument('--score_func', dest='score_func', default='conve', help='Score Function for Link prediction')
    parser.add_argument('--opn', dest='opn', default='corr', help='Composition Operation to be used in CompGCN')
    parser.add_argument('--batch_size', dest='batch_size', default=128, type=int, help='Batch size for GAT')
    parser.add_argument('--gamma', type=float, default=40.0, help='Margin')
    parser.add_argument('--gpu', type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('--epoch', dest='max_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--l2', type=float, default=0.0, help='L2 regularization for the optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='Starting Learning Rate')
    parser.add_argument('--lbl_smooth', dest='lbl_smooth', type=float, default=0.0, help='Label smoothing; set to 0 for contrastive loss')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of processes to construct batches')
    parser.add_argument('--seed', dest='seed', default=2023, type=int, help='Seed for randomization')
    parser.add_argument('--bias', dest='bias', default=0, type=int, help='Whether to use bias in the model')
    parser.add_argument('--num_bases', dest='num_bases', default=-1, type=int, help='Number of basis relation vectors to use')
    parser.add_argument('--init_dim', dest='init_dim', default=200, type=int, help='Initial dimension size for entities and relations')
    parser.add_argument('--gcn_dim', dest='gcn_dim', default=200, type=int, help='Number of hidden units in GCN')
    parser.add_argument('--embed_dim', dest='embed_dim', default=200, type=int, help='Embedding dimension to give as input to score function')
    parser.add_argument('--gcn_layer', dest='gcn_layer', default=2, type=int, help='Number of GCN Layers to use')
    parser.add_argument('--gcn_drop', dest='dropout', default=0.1, type=float, help='Dropout to use in GCN Layer')
    parser.add_argument('--hid_drop', dest='hid_drop', default=0.3, type=float, help='Dropout after GCN')
    parser.add_argument('--pretrained_emb', dest='pretrained_emb', default=0, type=int, help='')
    parser.add_argument('--num_of_layers', dest='num_of_layers', default=2, type=int, help='')
    parser.add_argument('--num_heads_per_layer', dest='num_heads_per_layer', nargs='*', default=[8, 1], type=int, help='')
    parser.add_argument('--add_skip_connection', dest='add_skip_connection', default=0, type=int, help='')
    parser.add_argument('--bias_gat', dest='bias_gat', default=1, type=int, help='')
    parser.add_argument('--dropout_gat', dest='dropout_gat', default=0.5, type=float, help='')
    parser.add_argument('--num_features_per_layer', dest='num_features_per_layer', nargs='*', default=[200, 25, 200], type=int, help='')
    parser.add_argument('--hid_drop2', dest='hid_drop2', default=0.3, type=float, help='ConvE: Hidden dropout')
    parser.add_argument('--feat_drop', dest='feat_drop', default=0.3, type=float, help='ConvE: Feature Dropout')
    parser.add_argument('--k_w', dest='k_w', default=10, type=int, help='ConvE: k_w')
    parser.add_argument('--k_h', dest='k_h', default=20, type=int, help='ConvE: k_h')
    parser.add_argument('--num_filt', dest='num_filt', default=200, type=int, help='ConvE: Number of filters in convolution')
    parser.add_argument('--ker_sz', dest='ker_sz', default=7, type=int, help='ConvE: Kernel size to use')
    parser.add_argument('--sage_dropout_rate', default=0.25, type=float, help='')
    parser.add_argument('--sage_kernel_size', default=5, type=int, help='')
    parser.add_argument('--sage_gc1_emb_size', default=200, type=float, help='')
    parser.add_argument('--sage_input_drop', default=0.0, type=float, help='')
    parser.add_argument('--sage_hidden_drop', default=0.2, type=float, help='')
    parser.add_argument('--sage_feature_map_drop', default=0.2, type=float, help='')
    parser.add_argument('--pos_sample_num', type=int, default=1, help='positive sample num')
    parser.add_argument('--neg_sample_num', type=int, default=100, help='negtive sample num')
    parser.add_argument('--subgraph_loss_rate', type=float, default=0.0, help='loss_rate')
    parser.add_argument('--node_loss_rate', type=float, default=0.001, help='loss_rate')
    parser.add_argument('--loss_rate', type=float, default=1.0, help='loss_rate')
    parser.add_argument('--use_pretrain', type=int, default=0, help='whether use pretrain')
    parser.add_argument('--share_kvq', type=int, default=0, help='share kvq weight')
    parser.add_argument('--model_nodetype', type=int, default=0, help='model nodetype')
    parser.add_argument('--model_relationtype', type=int, default=1, help='model relationtype')
    parser.add_argument('--mode_norm', type=str, default='BN', help='norm layer')
    parser.add_argument('--mode_activation', type=str, default='tanh', choices=['tanh', 'gelu'], help='Activation function')
    parser.add_argument('--mode_skip', type=str, default='dense', choices=['dense', 'residual'], help='Connection type between consecutive layer embeddings')
    parser.add_argument('--train_neg_num', type=int, default=200, help='Number of training negatives for contrastive loss')
    parser.add_argument('--direction', type=int, default=1, help='Whether the graph is directed')
    parser.add_argument('--node_num_bases', type=int, default=5, help='base metric num')
    parser.add_argument('--re_num_bases', type=int, default=5, help='base metric num')
    parser.add_argument('--lgtn_num_bases', type=int, default=0, help='base metric num')
    parser.add_argument('--style', type=str, default='mean', help='style of link for node emb')
    parser.add_argument('--local_rank', type=int, default=0, help='node rank for distributed training')
    parser.add_argument('--use_node_pretrain', type=int, default=0, help='whether use node pretrain')
    parser.add_argument('--use_relation_pretrain', type=int, default=0, help='whether use relation pretrain')
    parser.add_argument('--use_nodetype', type=int, default=0, help='Use node types; set to 1 only when node-type files are available')
    parser.add_argument('--use_relationtype', type=int, default=0, help='whether use relation type')
    parser.add_argument('--use_test', type=int, default=0, help='whether be used in test')
    parser.add_argument('--test_name', type=str, default='', help='test name')
    parser.add_argument('--save_path', type=str, default=None, help='load model from save path')
    parser.add_argument('--test', type=int, default=0, help='whether be used in testdata')
    parser.add_argument('--model_dir', type=str, default='fb15k237_model_save', help='The address for storing the models and optimization results.')
    parser.add_argument('--task_name', type=str, default='LP', help='The name of the stored models and optimization results.')
    parser.add_argument('--logdir', dest='log_dir', default='fb15k237_log', help='Log directory')
    parser.add_argument('--config', dest='config_dir', default='config', help='Config directory')
    parser.add_argument('--loss_function', type=str, default='bcel_loss', choices=['contrastive_loss', 'mask_softmax', 'mix_loss', 'bcel_loss'], help='choose loss function')
    parser.add_argument('--mix_loss_mod', type=str, default='CLoss-MLoss', choices=['CLoss-MLoss', 'CLoss-BCELoss', 'MLoss-BCELoss'], help='choose loss function')
    parser.add_argument('--sample_mod', type=str, default='part', help='Full or partial sampling for contrastive learning')
    parser.add_argument('--stop_num', type=int, default=50, help='stop num of epoch')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adamw', 'adam', 'sgd', 'adagrad'], help='optimizer to use.')
    parser.add_argument('--change_lr', default=0, type=int, help='change lr')
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR', choices=['CyclicLR', 'OneCycleLR', 'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau', 'CosineAnnealingLR'], help='lr scheduler to use.')
    parser.add_argument('--clip', type=float, default=0.25, help='Gradient Norm Clipping')
    parser.add_argument('--conv_name', type=str, default='lgtnv3', choices=['rgat', 'ssg', 'gin', 'fusedgat', 'gps', 'transe', 'complex', 'distmult', 'rotate', 'gtn', 'rhcgat', 'rhcgcn', 'rahcgt', 'transe', 'distmult', 'complex', 'simple', 'origin_sage', 'lgtnv3', 'lgtnv2', 'lgtn', 'rgcnv3', 'rgcnv2', 'rel_dense_mthgcl2', 'gatv4', 'gatv3', 'hrgat', 'heat', 'eg', 'film', 'gcnv2', 'gatv2', 'transformer', 'fa', 'supergat', 'indrel_dense_mthgcl', 'rel_dense_mthgcl', 'norel_dense_mthgcl', 'mthgcl', 'dense_mthgcl', 'mtmrhgnn', 'dense_mtmrhgnn', 'hgcl', 'dense_hgcl', 'compgcn', 'origin_compgcn', 'sage', 'hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn', 'sagecn'], help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
    parser.add_argument('--n_hid', type=int, default=200, help='Number of hidden dimension')
    parser.add_argument('--in_dim', type=int, default=200, help='Number of input dimension')
    parser.add_argument('--n_heads', type=int, default=10, help='Number of attention head')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout ratio')
    parser.add_argument('--n_linear', type=int, default=0, help='Number of GNN linear layers')
    parser.add_argument('--dimrate_linear', type=int, default=2, help='Number of GNN linear layers of dim rate')
    parser.add_argument('--skip1', type=int, default=0, help='whether to use skip conection in dense_hgcl')
    parser.add_argument('--skip2', type=int, default=0, help='whether to use skip conection in dense_hgcl')
    parser.add_argument('--init_func', type=str, default='xavier_uniform', choices=['xavier_normal', 'xavier_uniform', 'get_uniform', 'get_normal'], help='init parameters function')
    parser.add_argument('--att', type=str, default='no_att', choices=['prior_att', 'adaptive_att', 'no_att'], help='whether to use prior attation')
    parser.add_argument('--type_mode', type=str, default='single', choices=['multiple', 'single'], help='mode of type')
    parser.add_argument('--temperature', default=0.2, type=float, help='Temperature used in softmax')
    parser.add_argument('--dynamic_temp', default=0, type=int, help='dynamic Temperature used in softmax')
    parser.add_argument('--tau_plus', default=0.1, type=float, help='Positive class priorx')
    parser.add_argument('--beta', default=0.6, type=float, help='concentration parameter')
    parser.add_argument('--estimator', default='easy', type=str, help='Choose loss function')
    parser.add_argument('--Q', default=1.0, type=float, help='pso and neg sample of rate')
    parser.add_argument('--store_emb', default=None, type=int, help='save embedding')
    parser.add_argument('--store_name', type=str, default='MINCL', help='store_name')
    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(local_rank)
    seed = 2024
    set_seed(seed)
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    args.device = device
    args.rank = local_rank
    print('local_rank:', local_rank)
    print('device:', device)
    if args.use_pretrain:
        if args.use_test:
            name = args.test_name + '_' + args.score_func + '_' + args.loss_function + '_' + args.dataset + '_' + 'CL' + args.conv_name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H_%M_%S')
        else:
            name = args.score_func + '_' + args.loss_function + '_' + args.dataset + '_' + 'CL' + args.conv_name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H_%M_%S')
    elif args.use_test:
        name = args.test_name + '_' + 'notext_' + args.score_func + '_' + args.loss_function + '_' + args.dataset + '_' + 'CL' + args.conv_name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H_%M_%S')
    else:
        name = 'notext_' + args.score_func + '_' + args.loss_function + '_' + args.dataset + '_' + 'CL' + args.conv_name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H_%M_%S')
    args.name = name
    if args.dynamic_temp:
        print('use dynamic_temp')
    model = Runner(args)
    if args.test != 1:
        print('train')
        model.fit()
    else:
        print('test')
        model.test()
    current_directory = os.path.dirname(os.path.abspath('__file__'))
    pt = os.path.join(current_directory, args.model_dir.replace('.', ''), args.task_name + '_' + args.name)
    logging.info('file_path: %s' % pt)
    print('file_path:', pt)
    localtime = time.asctime(time.localtime(time.time()))
    logging.info(f'Program end time: {localtime}')
    dist.destroy_process_group()
