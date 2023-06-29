"""
PyTorch Child-Sum Tree LSTM model

See Tai et al. 2015 https://arxiv.org/abs/1503.00075 for model description.
"""

import torch


class TreeLSTM(torch.nn.Module):
    '''PyTorch TreeLSTM model that implements efficient batching.
    '''
    def __init__(self, in_features, out_features,num_labels,bio_or_not=False):
        '''TreeLSTM class initializer

        Takes in int sizes of in_features and out_features and sets up model Linear network layers.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bio_or_not=bio_or_not
        self.learnnode=torch.nn.Linear(2*self.out_features,self.out_features,bias=False)
        self.learntype_node = torch.nn.Linear(2 * self.out_features,self.out_features, bias=False)
        self.learntype_forw = torch.nn.Linear(2 * self.out_features, self.out_features, bias=False)
        self.learntype_back = torch.nn.Linear(2 * self.out_features, self.out_features, bias=False)
        self.softmax = torch.nn.Softmax(-1)
        self.num_labels=num_labels
        # bias terms are only on the W layers for efficiency
        self.U_iou = torch.nn.Linear(self.out_features, 3 * self.out_features, bias=False)
        self.U_fe_iou=torch.nn.Linear(self.out_features, 3 * self.out_features)
        self.U_fe_bio_iou = torch.nn.Linear(self.out_features, 3 * self.out_features)
        self.U_fe_bio_iou_rev = torch.nn.Linear(self.out_features, 3 * self.out_features)
        self.U_bi_iou=torch.nn.Linear(2*self.out_features, 3 * self.out_features, bias=False)
        # f terms are maintained seperate from the iou terms because they involve sums over child nodes
        # while the iou terms do not
        self.W_l_f = torch.nn.Linear(self.in_features, self.out_features, bias=False)
        self.W_l_f_rev=torch.nn.Linear(self.in_features, self.out_features, bias=False)
        self.W_r_f_rev = torch.nn.Linear(self.in_features, self.out_features, bias=False)
        self.W_l_iou_rev = torch.nn.Linear(self.in_features, 3*self.out_features, bias=False)
        self.W_r_iou_rev = torch.nn.Linear(self.in_features, 3*self.out_features, bias=False)
        self.W_r_f = torch.nn.Linear(self.in_features, self.out_features, bias=False)
        self.U_f = torch.nn.Linear(self.out_features, self.out_features, bias=False)
        self.U_fe_f=torch.nn.Linear(self.out_features, self.out_features)
        self.U_fe_bio_f=torch.nn.Linear(self.out_features, self.out_features)
        self.device = torch.device("cuda:0")
    def forward(self, features, node_order, adjacency_list, edge_order,finlist,labels,eh,treerev,noderev,edgerev):
        '''Run TreeLSTM model on a tree data structure with node features
        Takes Tensors encoding node features, a tree node adjacency_list, and the order in which 
        the tree processing should proceed in node_order and edge_order.
        '''
        batchsize = node_order.shape[0]
        # Total number of nodes in every tree in the batch
        batch_size = node_order.shape[1]
        # Retrive device the model is currently loaded on to generate h, c, and h_sum result buffers
        # h and c states for every node in the batch
        h = torch.zeros(batchsize,batch_size, self.out_features).cuda()
        hrev = torch.zeros(batchsize, batch_size, self.out_features).cuda()
        outfornode = torch.zeros(batchsize, self.out_features).cuda()
        outfow = torch.zeros(batchsize, self.out_features).cuda()
        outbak = torch.zeros(batchsize, self.out_features).cuda()
        outtoatt= torch.zeros(batchsize,batch_size, 2*self.out_features).cuda()
        # outoftest=torch.zeros(batchsize,self.num_labels,self.out_features).cuda()
        c = torch.zeros(batchsize,batch_size, self.out_features).cuda()
        crev = torch.zeros(batchsize, batch_size, self.out_features).cuda()
        trainmyfeature = torch.clone(features)
        e_after_wight_node=self.learnnode(eh)
        e_after_wight_type_node=self.learntype_node(eh)
        e_after_wight_type_bak = self.learntype_back(eh)
        e_after_wight_type_forw = self.learntype_forw(eh)
        # if labels is not None:
        for i in range(batchsize):
                trainmyfeature[i],numofnode= self.getfeature(features[i],finlist[i],e_after_wight_node[i])

                if self.bio_or_not:
                    for n in range(node_order[i].max() + 1):
                        self._run_bio_lstm(n, h[i], c[i], trainmyfeature[i], node_order[i], adjacency_list[i], edge_order[i])

                        # self._run_rev_bio_lstm(n, hrev[i], crev[i], trainmyfeature[i], noderev[i], treerev[i], edgerev[i])
                else:
                    for n in range(node_order[i].max() + 1):
                        self._run_lstm(n, h[i], c[i], trainmyfeature[i], node_order[i], adjacency_list[i], edge_order[i])
                        # self.rev_run_lstm(n, hrev[i], crev[i], trainmyfeature[i], noderev[i], treerev[i],edgerev[i])
                # outtoatt[i] =torch.cat([h[i],trainmyfeature[i]],dim=-1)
                # outfornode[i]=self.getfinalout(trainmyfeature[i],numofnode,e_after_wight_type_node[i])
                outfow[i]=self.getfinalout(h[i],numofnode,e_after_wight_type_forw[i])
                # outbak[i]=self.getfinalout(hrev[i],numofnode, e_after_wight_type_bak[i])
        # torch.cat([outfow], dim=-1)
        return outfow
        # else:
        #     for i in range(batchsize):
        #         for labelsnum in range(self.num_labels):
        #             myfeatures, numofnode = self.getfeature(features[i], finlist[i],labernum=labelsnum)
        #             if self.bio_or_not:
        #                 for n in range(node_order[i].max() + 1):
        #                     self._run_bio_lstm(n, h[i], c[i], myfeatures, node_order[i], adjacency_list[i],
        #                                        edge_order[i])
        #             else:
        #                 for n in range(node_order[i].max() + 1):
        #                     self._run_lstm(n, h[i], c[i], myfeatures, node_order[i], adjacency_list[i], edge_order[i])
        #             outoftest[i,labelsnum] = self.getfinalout(h[i], numofnode, labernum=labelsnum)
        #     return outoftest
    def getfeature(self,features,finlist,labeltensor):

        myfeature=torch.zeros(features.size(0),features.size(1)).cuda()
        a=0
        for i in range(len(finlist)):
            if finlist[i][0] == -1:
                    break
            else:
                    rep = features[finlist[i][0]:finlist[i][1] + 1]
                    att_score = (rep * labeltensor).sum(-1)
                    softmax_att_score = self.softmax(att_score)
                    bag_rep_3dim = (softmax_att_score.unsqueeze(-1) * rep).sum(0)
                    myfeature[i] = bag_rep_3dim
                    a = a + 1
        return   myfeature,a
    def getfinalout(self,h,numofnode,labeltensor):
        rep = h[0:numofnode]
        rep=torch.clone(rep)
        att_score = (rep * labeltensor).sum(-1)
        softmax_att_score = self.softmax(att_score)
        bag_rep_3dim = (softmax_att_score.unsqueeze(-1) * rep).sum(0)
        outfeature = bag_rep_3dim
        return outfeature

    def _run_lstm(self, iteration, h, c, features, node_order, adjacency_list, edge_order):
        '''Helper function to evaluate all tree nodes currently able to be evaluated.
        '''
        # N is the number of nodes in the tree
        # n is the number of nodes to be evaluated on in the current iteration
        # E is the number of edges in the tree
        # e is the number of edges to be evaluated on in the current iteration
        # F is the number of features in each node
        # M is the number of hidden neurons in the network

        # node_order is a tensor of size N x 1
        # edge_order is a tensor of size E x 1
        # features is a tensor of size N x F
        # adjacency_list is a tensor of size E x 2
        # node_mask is a tensor of size N x 1
        node_mask = node_order == iteration

        # edge_mask is a tensor of size E x 1
        edge_mask = edge_order == iteration

        # x is a tensor of size n x F

        # At iteration 0 none of the nodes should have children
        # Otherwise, select the child nodes needed for current iteration
        # and sum over their hidden states
        if iteration == 0:
            x=features[node_mask,:]

            iou = self.U_fe_iou(x)
        else:
            # adjacency_list is a tensor of size e x 2

            adjacency_list = adjacency_list[edge_mask, :]

            # parent_indexes and child_indexes are tensors of size e x 1
            # parent_indexes and child_indexes contain the integer indexes needed to index into
            # the feature and hidden state arrays to retrieve the data for those parent/child nodes.
            parent_indexes = adjacency_list[:, 1]

            child_indexes = adjacency_list[:, 0]
            # child_h and child_c are tensors of size e x 1
            child_h = h[child_indexes, :]
            child_c = c[child_indexes, :]

            # Add child hidden states to parent offset locations
            _, child_counts = torch.unique_consecutive(parent_indexes, return_counts=True)

            child_counts = tuple(child_counts)

            parent_children = torch.split(child_h, child_counts)

            parent_list = [item.sum(0) for item in parent_children]
            h_sum = torch.stack(parent_list)
            iou = self.U_iou(h_sum)+self.U_fe_iou(features[node_mask, :])

        # i, o and u are tensors of size n x M
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i = torch.sigmoid(i) #修改，考虑双关系
        o = torch.sigmoid(o)#修改，考虑当前关系
        u = torch.relu(u)

        # At iteration 0 none of the nodes should have children
        # Otherwise, calculate the forget states for each parent node and child node
        # and sum over the child memory cell states
        if iteration == 0:
            c[node_mask, :] = i * u
        else:
            # f is a tensor of size e x M
            # print(features[parent_indexes, :])
            f = self.U_f(child_h)+self.U_fe_f(features[parent_indexes, :])
            #+features[parent_indexes, :]

            f = torch.sigmoid(f)
            # fc is a tensor of size e x M
            fc= f * child_c
            # Add the calculated f values to the parent's memory cell state
            fc_new = torch.split(fc, child_counts)
            parent_list = [item.sum(0) for item in fc_new]

            c_sum = torch.stack(parent_list)
            c[node_mask, :] = i * u + c_sum

        h[node_mask, :] = o * torch.tanh(c[node_mask])
    def _run_bio_lstm(self, iteration, h, c, features, node_order, adjacency_list, edge_order):
        node_mask = node_order == iteration

        # edge_mask is a tensor of size E x 1
        edge_mask = edge_order == iteration

        # x is a tensor of size n x F


        # At iteration 0 none of the nodes should have children
        # Otherwise, select the child nodes needed for current iteration
        # and sum over their hidden states
        if iteration == 0:
            x = features[node_mask, :]

            iou = self.U_fe_bio_iou(x)

        else:
            # adjacency_list is a tensor of size e x 2
            adjacency_list = adjacency_list[edge_mask, :]

            # parent_indexes and child_indexes are tensors of size e x 1
            # parent_indexes and child_indexes contain the integer indexes needed to index into
            # the feature and hidden state arrays to retrieve the data for those parent/child nodes.
            parent_indexes = adjacency_list[:, 1]

            child_indexes = adjacency_list[:, 0]

            # child_h and child_c are tensors of size e x 1
            child_h = h[child_indexes, :]

            child_c = c[child_indexes, :]

            # Add child hidden states to parent offset locations
            _, child_counts1 = torch.unique_consecutive(parent_indexes, return_counts=True)

            child_counts = tuple(child_counts1)

            parent_children1 = torch.split(child_h, child_counts)
            parent_children = [torch.cat([item[0], item[1]], dim=0)if item.size(0)==2 else torch.cat([item[0], item[0]], dim=0) for item in parent_children1 ]

            iou= [self.U_bi_iou(item)if child_counts1[i]==2 else self.U_bi_iou(item)/2 for i,item in enumerate(parent_children) ]
            iou=torch.stack(iou)
            iou=iou+self.U_fe_bio_iou(features[node_mask, :])

        # i, o and u are tensors of size n x M
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)



        if iteration == 0:
            c[node_mask, :] = i * u
        else:
            # f is a tensor of size e x M
            f = [[self.W_l_f(item[0]),self.W_r_f(item[1])] if item.size(0) == 2 else [(self.W_l_f(item[0])+self.W_r_f(item[0]))/2] for item in parent_children1]
            a=[]
            for  item in f:
                a=a+item
            f=torch.stack(a)
            f=f+self.U_fe_bio_f(features[parent_indexes, :])
            # fc is a tensor of size e x M
            fc = f * child_c
            # Add the calculated f values to the parent's memory cell state
            parent_children = torch.split(fc, child_counts)
            parent_list = [item.sum(0) for item in parent_children]

            c_sum = torch.stack(parent_list)
            c[node_mask, :] = i * u + c_sum

        h[node_mask, :] = o * torch.tanh(c[node_mask])
    def _run_rev_bio_lstm(self, iteration, h, c, features, node_order, adjacency_list, edge_order):

        node_mask = node_order == iteration

        # edge_mask is a tensor of size E x 1
        edge_mask = edge_order == iteration

        # x is a tensor of size n x F


        # At iteration 0 none of the nodes should have children
        # Otherwise, select the child nodes needed for current iteration
        # and sum over their hidden states
        if iteration == 0:
            x = features[node_mask, :]

            iou = self.U_fe_bio_iou_rev(x)

        else:
            # adjacency_list is a tensor of size e x 2
            adjacency_list = adjacency_list[edge_mask, :]

            # parent_indexes and child_indexes are tensors of size e x 1
            # parent_indexes and child_indexes contain the integer indexes needed to index into
            # the feature and hidden state arrays to retrieve the data for those parent/child nodes.
            parent_indexes = adjacency_list[:, 1]

            child_indexes = adjacency_list[:, 0]

            # child_h and child_c are tensors of size e x 1
            child_h = h[child_indexes, :]

            child_c = c[child_indexes, :]

            # Add child hidden states to parent offset locations
            _, child_counts1 = torch.unique_consecutive(child_indexes, return_counts=True)

            child_counts = tuple(child_counts1)

            parent_children1 = torch.split(child_h, child_counts)
            iou = [[self.W_l_iou_rev(item[0]), self.W_r_iou_rev(item[1])] if item.size(0) == 2 else [
                (self.W_l_iou_rev(item[0]) + self.W_r_iou_rev(item[0])) / 2] for item in parent_children1]
            a = []
            for item in iou:
                a = a + item
            iou = torch.stack(a)

            iou=iou+self.U_fe_bio_iou_rev(features[node_mask, :])

        # i, o and u are tensors of size n x M
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)

        u = torch.relu(u)

        # At iteration 0 none of the nodes should have children
        # Otherwise, calculate the forget states for each parent node and child node
        # and sum over the child memory cell states
        if iteration == 0:
            c[node_mask, :] = i * u
        else:
            # f is a tensor of size e x M
            f = [[self.W_l_f_rev(item[0]),self.W_r_f_rev(item[1])] if item.size(0) == 2 else [(self.W_l_f_rev(item[0])+self.W_r_f_rev(item[0]))/2] for item in parent_children1]
            a=[]
            for  item in f:
                a=a+item
            f=torch.stack(a)
            f=f+self.U_fe_bio_f(features[parent_indexes, :])
            # fc is a tensor of size e x M
            fc = f * child_c
            # Add the calculated f values to the parent's memory cell state


            c[node_mask, :] = i * u + fc

        h[node_mask, :] = o * torch.relu(c[node_mask])
