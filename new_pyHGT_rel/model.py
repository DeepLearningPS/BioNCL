from .conv import *
from torch_geometric.nn.inits import glorot, uniform
from .helper import *
import numpy as np
import time
from torch.nn.init import xavier_normal_, xavier_uniform_

class Classifier(nn.Module):

    def __init__(self, n_hid, n_out):
        super(Classifier, self).__init__()
        self.n_hid = n_hid
        self.n_out = n_out
        self.linear = nn.Linear(n_hid, n_out)

    def forward(self, x):
        tx = self.linear(x)
        return torch.log_softmax(tx.squeeze(), dim=-1)

    def __repr__(self):
        return '{}(n_hid={}, n_out={})'.format(self.__class__.__name__, self.n_hid, self.n_out)

class Matcher(nn.Module):

    def __init__(self, args, n_hid):
        super(Matcher, self).__init__()
        self.p = args
        self.n_hid = n_hid
        self.sqrt_hd = math.sqrt(n_hid)
        self.x_bn = torch.nn.BatchNorm1d(self.p.num_ent)
        self.x_ln = torch.nn.LayerNorm(self.p.num_ent)
        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
        if self.p.score_func in ['conve', 'conve_complex', 'conve_ditmult']:
            self.bn0 = torch.nn.BatchNorm2d(1)
            self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
            self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)
            self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
            self.hidden_drop2 = torch.nn.Dropout(self.p.hid_drop2)
            self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
            self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)
            flat_sz_h = int(2 * self.p.k_w) - self.p.ker_sz + 1
            flat_sz_w = self.p.k_h - self.p.ker_sz + 1
            self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt
            self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)
        self.x_bn = torch.nn.BatchNorm1d(self.p.num_ent)
        self.x_ln = torch.nn.LayerNorm(self.p.num_ent)

    def transe(self, sub_emb, rel_emb, all_ent):
        obj_emb = sub_emb + rel_emb
        x = self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
        x += self.bias.expand_as(x)
        if self.p.loss_function == 'contrastive_loss':
            score = x
            score = self.x_ln(score)
        else:
            score = x
        return score

    def distmult(self, sub_emb, rel_emb, all_ent, mode=None):
        obj_emb = sub_emb * rel_emb
        x = torch.mm(obj_emb, all_ent.transpose(1, 0))
        if mode != None:
            return x
        x += self.bias.expand_as(x)
        if self.p.loss_function == 'contrastive_loss':
            score = x
            score = self.x_ln(score)
        else:
            score = x
        return score

    def complex(self, sub_emb, rel_emb, all_ent, mode=None):
        heads = sub_emb
        tails = all_ent
        rels = rel_emb
        heads_re, heads_im = torch.chunk(heads, chunks=2, dim=-1)
        tails_re, tails_im = torch.chunk(tails, chunks=2, dim=-1)
        rels_re, rels_im = torch.chunk(rels, chunks=2, dim=-1)
        x = torch.mm(rels_re * heads_re, tails_re.transpose(1, 0)) + torch.mm(rels_re * heads_im, tails_im.transpose(1, 0)) + torch.mm(rels_im * heads_re, tails_im.transpose(1, 0)) - torch.mm(rels_im * heads_im, tails_re.transpose(1, 0))
        if mode != None:
            return x
        x += self.bias.expand_as(x)
        if self.p.loss_function == 'contrastive_loss':
            score = x
            score = self.x_ln(score)
        else:
            score = x
        return score

    def simple(self, sub_emb, rel_emb, all_ent, mode=None):
        heads = sub_emb
        tails = all_ent
        rels = rel_emb
        heads_h, heads_t = torch.chunk(heads, chunks=2, dim=-1)
        tails_h, tails_t = torch.chunk(tails, chunks=2, dim=-1)
        rel_a, rel_b = torch.chunk(rels, chunks=2, dim=-1)
        x = (torch.mm(heads_h * rel_a, tails_t.transpose(1, 0)) + torch.mm(heads_t * rel_b, tails_h.transpose(1, 0))) / 2
        if mode != None:
            return x
        x += self.bias.expand_as(x)
        if self.p.loss_function == 'contrastive_loss':
            score = x
            score = self.x_ln(score)
        else:
            score = x
        return score

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.p.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.p.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.p.k_w, self.p.k_h))
        return stack_inp

    def conve_old(self, sub_emb, rel_emb, all_ent):
        stk_inp = self.concat(sub_emb, rel_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)
        if self.p.loss_function == 'bcel_loss':
            score_func = x
        else:
            score_func = x
        return score_func

    def conve(self, sub_emb, rel_emb, all_ent):
        stk_inp = self.concat(sub_emb, rel_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)
        if self.p.loss_function == 'contrastive_loss':
            score = x
            score = self.x_ln(score)
        else:
            score = x
        return score

    def conve_complex(self, sub_emb, rel_emb, all_ent):
        stk_inp = self.complex(sub_emb, rel_emb, all_ent, mode='conve_complex')
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)
        if self.p.loss_function == 'bcel_loss':
            score_func = x
        else:
            score_func = x
        return score_func

    def conve_distmult(self, sub_emb, rel_emb, all_ent):
        stk_inp = self.distmult(sub_emb, rel_emb, all_ent, mode='conve_complex')
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)
        if self.p.loss_function == 'bcel_loss':
            score_func = x
        else:
            score_func = x
        return score_func

    def conve_transe_old(self, sub_emb, rel_emb, all_ent):
        e1_embedded_all = all_ent
        e1_embedded = sub_emb
        rel_embedded = rel_emb
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(self.p.batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        if self.p.loss_function == 'bcel_loss':
            score_func = x
        else:
            score_func = x
        return score_func

    def conve_transe(self, sub_emb, rel_emb, all_ent):
        e1_embedded_all = all_ent
        e1_embedded = sub_emb
        rel_embedded = rel_emb
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(self.p.batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        if self.p.loss_function == 'bcel_loss':
            score_func = x
        else:
            score_func = x
        score = score_func / self.p.embed_dim
        score_norm = torch.norm(score, p=2, dim=-1, keepdim=True)
        score = score / score_norm
        score = self.x_bn(score)
        return score

    def forward(self, x, r, y):
        sub_emb = x
        all_emb = y
        rel_emb = r
        if self.p.score_func == 'transe':
            res = self.transe(sub_emb, rel_emb, all_emb)
        elif self.p.score_func == 'distmult':
            res = self.distmult(sub_emb, rel_emb, all_emb)
        elif self.p.score_func == 'complex':
            res = self.complex(sub_emb, rel_emb, all_emb)
        elif self.p.score_func == 'simple':
            res = self.simple(sub_emb, rel_emb, all_emb)
        elif self.p.score_func == 'conve':
            res = self.conve(sub_emb, rel_emb, all_emb)
        elif self.p.score_func == 'conve_complex':
            res = self.conve_complex(sub_emb, rel_emb, all_emb)
        elif self.p.score_func == 'conve_transe':
            res = self.conve_transe(sub_emb, rel_emb, all_emb)
        else:
            raise ValueError('Choose one scoring function: transe, distmult, complex, simple, conve, conve_complex, or conve_transe')
        return res

    def __repr__(self):
        return '{}(n_hid={})'.format(self.__class__.__name__, self.n_hid)

class GNN(nn.Module):

    def __init__(self, edge_index, edge_type, rel_attribute, node_attribute, type_matrix, adj, params=None, prev_norm=True, last_norm=True):
        super(GNN, self).__init__()
        self.p = params
        self.num_types = self.p.num_nodetype
        self.num_relations = self.p.num_rel
        self.num_ent = self.p.num_ent
        self.in_dim = self.p.in_dim
        self.n_hid = self.p.n_hid
        self.n_layers = self.p.n_layers
        self.n_heads = self.p.n_heads
        self.conv_name = self.p.conv_name
        self.edge_index = edge_index.cuda()
        self.edge_type = edge_type.cuda()
        self.rel_attribute = rel_attribute
        self.node_attribute = node_attribute
        self.type_matrix = type_matrix.cuda()
        self.type_count = torch.ones_like(self.type_matrix).cuda()
        self.adj = adj
        self.adj[0] = self.adj[0].cuda()
        self.adj[1] = self.adj[1].cuda()
        self.gcs = nn.ModuleList()
        self.drop = nn.Dropout(self.p.dropout)
        self.comp_drop = torch.nn.Dropout(self.p.hid_drop)
        self.comp_feature_drop = torch.nn.Dropout(self.p.feat_drop)
        self.models = ['lgtnv2', 'lgtn', 'rgcnv3', 'rgcnv2', 'rel_dense_mthgcl2', 'gatv3', 'gatv4', 'hrgat', 'compgcn', 'sage', 'mtmrhgnn', 'dense_mtmrhgnn', 'hgcl', 'dense_hgcl', 'mthgcl', 'dense_mthgcl', 'norel_dense_mthgcl', 'rel_dense_mthgcl', 'indrel_dense_mthgcl']
        self.rel_models = ['gtn', 'lgtnv3', 'lgtnv2', 'lgtn', 'rgcnv3', 'rgcnv2', 'rel_dense_mthgcl', 'rel_dense_mthgcl2', 'indrel_dense_mthgcl', 'gatv3', 'gatv4', 'hrgat']
        self.full_GNN = ['transe', 'complex', 'distmult', 'rotate']
        print('Number of entity types:', self.num_types)
        print('Number of relation types:', self.num_relations)
        if self.p.use_node_pretrain == 0:
            self.node_attribute = get_param((self.p.num_ent, self.in_dim)).cuda()
            self.node_adapt_ws = nn.Linear(self.in_dim, self.n_hid)
        else:
            self.node_attribute = self.node_attribute.cuda()
            self.node_adapt_ws = nn.Linear(self.node_attribute.size(1), self.n_hid)
        if self.p.use_relation_pretrain == 0:
            self.rel_attribute = get_param((self.num_relations, self.in_dim)).cuda()
            self.rel_adapt_ws = nn.Linear(self.in_dim, self.n_hid)
        else:
            self.rel_attribute = self.rel_attribute.cuda()
            self.rel_adapt_ws = nn.Linear(self.rel_attribute.size(1), self.n_hid)
        for nl in range(self.n_layers - 1):
            self.gcs.append(GeneralConv(params, self.conv_name, self.n_hid, self.n_hid, self.num_types, self.num_relations, self.n_heads, self.p.dropout, use_norm=prev_norm))
        self.gcs.append(GeneralConv(params, self.conv_name, self.n_hid, self.n_hid, self.num_types, self.num_relations, self.n_heads, self.p.dropout, use_norm=last_norm))
        if self.conv_name in ['sage', 'origin_sage'] or (self.conv_name not in self.rel_models and self.conv_name != 'norel_dense_mthgcl'):
            self.bn = nn.ModuleList()
            self.drop = nn.Dropout(self.p.dropout)
            for l in range(self.n_layers):
                self.bn.append(torch.nn.BatchNorm1d(self.n_hid))

    def reset_parameters(self):
        size = self.p.node_num_bases * self.in_dim
        if self.p.init_func == 'xavier_normal':
            xavier_normal_(self.basis.data)
            xavier_normal_(self.att.data)
            if self.p.bias != 0:
                xavier_normal_(self.bias.data)
                xavier_normal_(self.bias_att.data)
        elif self.p.init_func == 'xavier_uniform':
            xavier_uniform_(self.basis.data)
            xavier_uniform_(self.att.data)
            if self.p.bias != 0:
                get_uniform(size, self.bias)
                get_uniform(size, self.bias_att)
        elif self.p.init_func == 'get_uniform':
            get_uniform(size, self.basis)
            get_uniform(size, self.att)
            get_uniform(size, self.bias)
            get_uniform(size, self.bias_att)
        elif self.p.init_func == 'get_normal':
            get_normal(self.basis)
            get_normal(self.att)
            get_normal(self.bias)
            get_normal(self.bias_att)
        else:
            raise Exception('input one of xavier_normal/xavier_uniform/get_uniform/get_normal')

    def forward(self, x_id, r_id):
        node_res = self.node_attribute
        r = self.rel_attribute
        if self.conv_name in ['sage', 'origin_sage']:
            for gcc, bn in zip(self.gcs, self.bn):
                node_res = gcc(node_res, self.type_matrix, self.type_count, self.edge_index, self.edge_type, r, self.adj)
                node_res = bn(node_res)
                node_res = F.tanh(node_res)
                node_res = self.drop(node_res)
            x_emb = node_res[x_id]
            y_emb = node_res
            r_emb = r[r_id]
        elif self.conv_name in ['compgcn', 'origin_compgcn']:
            for gcc in self.gcs:
                node_res, r = gcc(node_res, self.type_matrix, self.type_count, self.edge_index, self.edge_type, r, self.adj)
                node_res = self.drop(node_res)
            x_emb = node_res[x_id]
            y_emb = node_res
            r_emb = r[r_id]
        elif self.conv_name in self.rel_models:
            for gcc in self.gcs:
                node_res, r = gcc(node_res, self.type_matrix, self.type_count, self.edge_index, self.edge_type, r, self.adj)
            x_emb = node_res[x_id]
            y_emb = node_res
            r_emb = r[r_id]
        elif self.conv_name == 'norel_dense_mthgcl':
            for gcc in self.gcs:
                node_res = gcc(node_res, self.type_matrix, self.type_count, self.edge_index, self.edge_type, r, self.adj)
            x_emb = node_res[x_id]
            y_emb = node_res
            r_emb = r[r_id]
        elif self.conv_name == 'fusedgat':
            csr, csc, perm = FusedGATConv.to_graph_format(self.edge_index)
            for gcc, bn in zip(self.gcs, self.bn):
                node_res = gcc(node_res, self.type_matrix, self.type_count, self.edge_index, self.edge_type, r, self.adj, csr, csc, perm)
                node_res = bn(node_res)
                node_res = F.tanh(node_res)
                node_res = self.drop(node_res)
            x_emb = node_res[x_id]
            y_emb = node_res
            r_emb = r[r_id]
        elif self.conv_name in self.full_GNN:
            for gcc, bn in zip(self.gcs, self.bn):
                node_res = gcc(node_res, self.type_matrix, self.type_count, self.edge_index, self.edge_type, r, self.adj)
                node_res = bn(node_res)
                node_res = F.tanh(node_res)
                node_res = self.drop(node_res)
            x_emb = node_res[x_id]
            y_emb = node_res
            r_emb = r[r_id]
        else:
            for gcc, bn in zip(self.gcs, self.bn):
                node_res = gcc(node_res, self.type_matrix, self.type_count, self.edge_index, self.edge_type, r, self.adj)
                node_res = bn(node_res)
                node_res = F.tanh(node_res)
                node_res = self.drop(node_res)
            x_emb = node_res[x_id]
            y_emb = node_res
            r_emb = r[r_id]
        if self.p.store_emb != None:
            return (x_emb, y_emb, r_emb, x_emb, r_emb)
        else:
            return (x_emb, r_emb, y_emb)
