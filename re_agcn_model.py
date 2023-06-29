import copy
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from model.bert import BertPreTrainedModel, BertModel
from model.agcn import TypeGraphConvolution
from tree_lstm import TreeLSTM
import numpy as np
np.set_printoptions(threshold=np.inf)
from torch.cuda import amp
last_memory=0
def get_memory():
    global last_memory
    last_memory = torch.cuda.memory_allocated() / 1024 / 1024
    return last_memory
def get_memory_diff():
    last = last_memory
    total = get_memory()
    return total - last, total
class ReAgcn(BertPreTrainedModel):
    def __init__(self, config):
        super(ReAgcn, self).__init__(config)
        self.bert=BertModel(self.config)
        self.dep_type_embedding = nn.Embedding(config.type_num, config.hidden_size, padding_idx=0)

        self.tree_tpye_embedding = nn.Embedding(config.tree_labels_len,config.hidden_size, padding_idx=0)
        # gcn_layer = TypeGraphConvolution(config.hidden_size, config.hidden_size)
        self.gcnfistlayer=TypeGraphConvolution(config.hidden_size, config.hidden_size)
        self.gcn_layer =TypeGraphConvolution(config.hidden_size, config.hidden_size)
        self.gcnn=TypeGraphConvolution(config.hidden_size, config.hidden_size)
        # self.gcn_layer = nn.ModuleList([copy.deepcopy(gcn_layer) for _ in range(config.num_gcn_layers)])
        self.ensemble_linear = nn.Linear(1, config.num_gcn_layers)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*6, config.num_labels)
        self.sent = nn.Linear(config.hidden_size, config.hidden_size*2)
        self.apply(self.init_bert_weights)
        self.num_labels=config.num_labels
        self.tree = TreeLSTM(config.hidden_size, config.hidden_size,config.num_labels)
        self.tree1 = TreeLSTM(config.hidden_size, config.hidden_size, config.num_labels)
        self.softmax=nn.Softmax(-1)

    def valid_filter(self, sequence_output, valid_ids):
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=sequence_output.dtype,
                                   device=sequence_output.device)
        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            valid_output[i][:temp.size(0)] = temp
        return valid_output

    def max_pooling(self, sequence, e_mask):
        entity_output = sequence * torch.stack([e_mask] * sequence.shape[-1], 2) + torch.stack(
            [(1.0 - e_mask) * -1000.0] * sequence.shape[-1], 2)
        entity_output = torch.max(entity_output, -2)[0]
        return entity_output.type_as(sequence)

    def extract_entity(self, sequence, e_mask):
        return self.max_pooling(sequence, e_mask)
    def get_attention(self, val_out, dep_embed, adj,pool):
        batch_size, max_len, feat_dim = val_out.shape

        val_us = val_out.unsqueeze(dim=2)
        val_us = checkpoint(val_us.repeat,1,1,max_len,1)

        val_us = checkpoint(torch.cat,(val_us, dep_embed), -1)
        val_us = (val_us * val_us.transpose(1,2))
        # pool=self.sent(pool)
        # for i in range(batch_size):
        #  val_us[i]=pool[i]+val_us[i]

        # val_us=val_us+dep_embed

        val_us = torch.sum(val_us, dim=-1)
        val_us= val_us / feat_dim ** 0.5
        val_us = torch.exp(val_us)
        val_us = torch.mul(val_us.float(), adj.float())

        sum_attention_score = torch.sum(val_us, dim=-1).unsqueeze(dim=-1).repeat(1,1,max_len)
        return torch.div(val_us, sum_attention_score + 1e-10)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, e1_mask=None, e2_mask=None,
                dep_adj_matrix=None, dep_type_matrix=None, valid_ids=None, tree=None, node_order=None, edge_order=None,finlist=None,treerev=None,noderev=None,edgerev=None,dep_type=None,dep_adj=None):
            sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                       False, True)

            if valid_ids is not None:
                sequence_output = self.valid_filter(sequence_output, valid_ids)
            else:
                sequence_output = sequence_output

            sequence_output = self.dropout(sequence_output)
            dep_adj_matrix = torch.clamp(dep_adj_matrix, 0, 1)

            typeemb=torch.sum(self.dep_type_embedding(dep_type),dim=-2)/2

            attention_score0 = self.get_attention(sequence_output, self.dep_type_embedding(dep_type_matrix),
                                                 dep_adj_matrix, pooled_output)

            attention_score1= self.get_attention(sequence_output, typeemb,
                                                  dep_adj, pooled_output)

            sequence_output0 = checkpoint(self.gcn_layer, sequence_output, attention_score0,
                                         self.dep_type_embedding(dep_type_matrix))

            sequence_output1 = checkpoint(self.gcnfistlayer, sequence_output, attention_score1,
                                          typeemb)

            # # for i, gcn_layer_module in enumerate(self.gcn_layer):
            # #     attention_score = self.get_attention(sequence_output, self.dep_type_embedding(dep_type_matrix), dep_adj_matrix,pooled_output)
            # #     sequence_output = checkpoint(gcn_layer_module,sequence_output, attention_score, self.dep_type_embedding(dep_type_matrix)
            # # sequence_output22 = checkpoint(self.gcnn, sequence_output, attention_score1,
            # #                               typeemb)
            e1_h=self.extract_entity(sequence_output0, e1_mask)
            e2_h=self.extract_entity(sequence_output0, e2_mask)

            e11_h = self.extract_entity(sequence_output1, e1_mask)
            e12_h = self.extract_entity(sequence_output1, e2_mask)
            eh0=torch.cat([e1_h, e2_h], dim=-1)

            eh1 = torch.cat([e11_h, e12_h], dim=-1)

            pooled_output0 = checkpoint(self.tree,sequence_output0, node_order, tree, edge_order,finlist,labels,eh0,treerev,noderev,edgerev)
            pooled_output1 = checkpoint(self.tree1, sequence_output1, node_order, tree, edge_order, finlist, labels, eh1,
                                        treerev, noderev, edgerev)
            pooled_output = torch.cat([pooled_output0,pooled_output1, e1_h, e2_h,e11_h,e12_h], dim=-1)
            pooled_output = self.dropout(pooled_output)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(self.classifier(pooled_output).view(-1, self.num_labels), labels.view(-1))
                return loss
            else:
                # newpooled = torch.zeros(pooled_output.size(0), self.num_labels, pooled_output.size(1)).cuda()
                # for i in range(self.num_labels):
                #    pooled_outputout = (pooled_output+pooled_output1[:,i,:]+self.extract_entity(sequence_output, e1_mask)+self.extract_entity(sequence_output, e2_mask))/4
                #    pooled_outputout = self.dropout(pooled_outputout)
                #    newpooled[:,i,:]=pooled_outputout

                return self.classifier(pooled_output)
