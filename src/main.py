# @Time    : 2019-08-10 11:20
# @Author  : Lee_zix
# @Email   : Lee_zix@163.com
# @File    : main.py
# @Software: PyCharm
"""
The entry of the KGEvolve
"""

import argparse
import itertools
import os
import sys
import time
import pickle
import torch.nn as nn
import dgl
import numpy as np
import torch
from tqdm import tqdm
import random
sys.path.append("..")
from rgcn import utils
from rgcn.utils import build_sub_graph
from src.rrgcn import RecurrentRGCN
from src.hyperparameter_range import hp_range
import torch.nn.modules.rnn
from collections import defaultdict
from rgcn.knowledge_graph import _read_triplets_as_list
# os.environ['KMP_DUPLICATE_LIB_OK']='True'


def test(model, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, all_ans_r_list, model_name, static_graph, mode):
    """
    :param model: model used to test
    :param history_list:    all input history snap shot list, not include output label train list or valid list
    :param test_list:   test triple snap shot list
    :param num_rels:    number of relations
    :param num_nodes:   number of nodes
    :param use_cuda:
    :param all_ans_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param all_ans_r_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param model_name:
    :param static_graph
    :param mode
    :return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r
    """
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []

    idx = 0
    if mode == "test":
        # test mode: load parameter form file
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    # do not have inverse relation in test input
    input_list = [snap for snap in history_list[-args.test_history_len:]]

    for time_idx, test_snap in enumerate(tqdm(test_list)):
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]
        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input = test_triples_input.to(args.gpu)
        test_triples, final_score, final_r_score = model.predict(history_glist, num_rels, static_graph, test_triples_input, use_cuda)

        mrr_filter_snap_r, mrr_snap_r, rank_raw_r, rank_filter_r = utils.get_total_rank(test_triples, final_r_score, all_ans_r_list[time_idx], eval_bz=1000, rel_predict=1)
        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)

        # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        # used to show slide results
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)

        # relation rank
        ranks_raw_r.append(rank_raw_r)
        ranks_filter_r.append(rank_filter_r)
        mrr_raw_list_r.append(mrr_snap_r)
        mrr_filter_list_r.append(mrr_filter_snap_r)

        # reconstruct history graph list
        if args.multi_step:
            if not args.relation_evaluation:    
                predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
            else:
                predicted_snap = utils.construct_snap_r(test_triples, num_nodes, num_rels, final_r_score, args.topk)
            if len(predicted_snap):
                input_list.pop(0)
                input_list.append(predicted_snap)
        else:
            input_list.pop(0)
            input_list.append(test_snap)
        idx += 1
    
    mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    mrr_raw_r = utils.stat_ranks(ranks_raw_r, "raw_rel")
    mrr_filter_r = utils.stat_ranks(ranks_filter_r, "filter_rel")
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r

def train_function(epochs, model, train_list, num_nodes, num_rels, use_cuda, static_graph, optimizer, model_name, valid_list, all_ans_list_valid, all_ans_list_r_valid, model_state_file):
    best_mrr = 0
    for epoch in range(epochs):
        model.train()
        losses = []
        losses_e = []
        losses_r = []
        losses_static = []

        idx = [_ for _ in range(len(train_list))]
        random.shuffle(idx)

        for train_sample_num in tqdm(idx):
            if train_sample_num == 0: continue
            output = train_list[train_sample_num:train_sample_num+1]
            if train_sample_num - args.train_history_len<0:
                input_list = train_list[0: train_sample_num]
            else:
                input_list = train_list[train_sample_num - args.train_history_len:
                                    train_sample_num]

            # generate history graph
            history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
            output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [torch.from_numpy(_).long() for _ in output]
            loss_e, loss_r, loss_static = model.get_loss(history_glist, output[0], static_graph, use_cuda)
            loss = args.task_weight*loss_e + (1-args.task_weight)*loss_r + loss_static

            losses.append(loss.item())
            losses_e.append(loss_e.item())
            losses_r.append(loss_r.item())
            losses_static.append(loss_static.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            optimizer.step()
            optimizer.zero_grad()

        print("Epoch {:04d} | Ave Loss: {:.4f} | entity-relation-static:{:.4f}-{:.4f}-{:.4f} Best MRR {:.4f} | Model {} "
                .format(epoch, np.mean(losses), np.mean(losses_e), np.mean(losses_r), np.mean(losses_static), best_mrr, model_name))

        # validation
        if epoch and epoch % args.evaluate_every == 0:
            mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
                                                                train_list, 
                                                                valid_list, 
                                                                num_rels, 
                                                                num_nodes, 
                                                                use_cuda, 
                                                                all_ans_list_valid, 
                                                                all_ans_list_r_valid, 
                                                                model_state_file, 
                                                                static_graph, 
                                                                mode="train")
            
            if not args.relation_evaluation:  # entity prediction evalution
                if mrr_filter > best_mrr:
                    best_mrr = mrr_filter
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
            else:
                if mrr_filter_r > best_mrr:
                    best_mrr = mrr_filter_r
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)

def train_distill_stage1(epochs, quality_filter, value_entity_filter, value_relation_filter, model, model_teacher, train_list, num_nodes, num_rels, use_cuda, static_graph, optimizer, model_name, valid_list, all_ans_list_valid, all_ans_list_r_valid, model_state_file):
    best_mrr = 0
    for epoch in range(epochs):
        model.train()
        losses = []
        losses_e = []
        losses_r = []
        losses_static = []

        idx = [_ for _ in range(len(train_list))]
        random.shuffle(idx)

        for train_sample_num in tqdm(idx):
            if train_sample_num == 0: continue
            output = train_list[train_sample_num:train_sample_num+1]
            if train_sample_num - args.train_history_len<0:
                input_list = train_list[0: train_sample_num]
            else:
                input_list = train_list[train_sample_num - args.train_history_len:
                                    train_sample_num]

            # generate history graph
            history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
            output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [torch.from_numpy(_).long() for _ in output]
            inverse_triples = output[0][:, [2, 1, 0]]
            inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
            all_triples = torch.cat([output[0], inverse_triples])
            all_triples = all_triples.to(args.gpu)
            scores_ob, score_rel = model_teacher.soft_label(history_glist, num_rels, static_graph, all_triples, use_cuda)
            quality_entity_score = quality_filter(torch.gather(scores_ob, 1, all_triples[:, 2].unsqueeze(-1)))
            value_entity_score = value_entity_filter(scores_ob)
            quality_relation_score = quality_filter(torch.gather(score_rel, 1, all_triples[:, 1].unsqueeze(-1)))
            value_relation_score = value_relation_filter(score_rel)
            loss_e, loss_r, loss_static = model.get_distillation_loss(history_glist, all_triples, static_graph, scores_ob, score_rel, quality_entity_score,
                                                                      value_entity_score, quality_relation_score, value_relation_score, use_cuda)
            loss = args.task_weight * loss_e + (1-args.task_weight) * loss_r + loss_static

            losses.append(loss.item())
            losses_e.append(loss_e.item())
            losses_r.append(loss_r.item())
            losses_static.append(loss_static.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            optimizer.step()
            optimizer.zero_grad()

        print("Epoch {:04d} | Ave Loss: {:.4f} | entity-relation-static:{:.4f}-{:.4f}-{:.4f} Best MRR {:.4f} | Model {} "
                .format(epoch, np.mean(losses), np.mean(losses_e), np.mean(losses_r), np.mean(losses_static), best_mrr, model_name))

        # validation
        if epoch and epoch % args.evaluate_every == 0:
            mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
                                                                train_list, 
                                                                valid_list, 
                                                                num_rels, 
                                                                num_nodes, 
                                                                use_cuda, 
                                                                all_ans_list_valid, 
                                                                all_ans_list_r_valid, 
                                                                model_state_file, 
                                                                static_graph, 
                                                                mode="train")
            
            if not args.relation_evaluation:  # entity prediction evalution
                if mrr_filter > best_mrr:
                    best_mrr = mrr_filter
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
            else:
                if mrr_filter_r > best_mrr:
                    best_mrr = mrr_filter_r
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)

def train_distill_stage2(epochs, value_entity_filter, value_relation_filter, model, model_teacher, train_list, num_nodes, num_rels, use_cuda, static_graph, optimizer, model_name, valid_list, all_ans_list_valid, all_ans_list_r_valid, model_state_file):
    best_mrr = 0
    for epoch in range(epochs):
        model.train()
        losses = []
        losses_e = []
        losses_r = []
        losses_static = []

        idx = [_ for _ in range(len(train_list))]
        random.shuffle(idx)

        for train_sample_num in tqdm(idx):
            if train_sample_num == 0: continue
            output = train_list[train_sample_num:train_sample_num+1]
            if train_sample_num - args.train_history_len<0:
                input_list = train_list[0: train_sample_num]
            else:
                input_list = train_list[train_sample_num - args.train_history_len:
                                    train_sample_num]

            # generate history graph
            history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
            output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [torch.from_numpy(_).long() for _ in output]
            inverse_triples = output[0][:, [2, 1, 0]]
            inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
            all_triples = torch.cat([output[0], inverse_triples])
            all_triples = all_triples.to(args.gpu)
            scores_ob, score_rel = model_teacher.soft_label(history_glist, num_rels, static_graph, all_triples, use_cuda)
            loss_e, loss_r, loss_static = model.get_distillation_loss_reverse(history_glist, all_triples, static_graph, scores_ob, score_rel,
                                                                      value_entity_filter, value_relation_filter, use_cuda)
            loss = args.task_weight * loss_e + (1-args.task_weight) * loss_r + loss_static

            losses.append(loss.item())
            losses_e.append(loss_e.item())
            losses_r.append(loss_r.item())
            losses_static.append(loss_static.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            optimizer.step()
            optimizer.zero_grad()

        print("Epoch {:04d} | Ave Loss: {:.4f} | entity-relation-static:{:.4f}-{:.4f}-{:.4f} Best MRR {:.4f} | Model {} "
                .format(epoch, np.mean(losses), np.mean(losses_e), np.mean(losses_r), np.mean(losses_static), best_mrr, model_name))

        # validation
        if epoch and epoch % args.evaluate_every == 0:
            mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
                                                                train_list, 
                                                                valid_list, 
                                                                num_rels, 
                                                                num_nodes, 
                                                                use_cuda, 
                                                                all_ans_list_valid, 
                                                                all_ans_list_r_valid, 
                                                                model_state_file, 
                                                                static_graph, 
                                                                mode="train")
            
            if not args.relation_evaluation:  # entity prediction evalution
                if mrr_filter > best_mrr:
                    best_mrr = mrr_filter
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
            else:
                if mrr_filter_r > best_mrr:
                    best_mrr = mrr_filter_r
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)

class quality_filter(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(quality_filter, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(input_size * 2, input_size * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(input_size * 4, input_size * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(input_size * 8, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.clone().detach().requires_grad_(True)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.as_tensor(self.layer4(out),dtype=torch.float32)
        return out

class value_filter(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(value_filter, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(input_size * 2, input_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(input_size // 2, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.clone().detach().requires_grad_(True)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.as_tensor(self.layer4(out),dtype=torch.float32)
        return out

def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    # load configuration for grid search the best configuration
    if n_hidden:
        args.n_hidden = n_hidden
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases
    print(args.dataset)
    # load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset)
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)

    num_nodes = data.num_nodes
    num_rels = data.num_rels

    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)

    # print("Sanity Check: stat name : {}".format(model_state_file_teacher))
    # print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    if args.add_static_graph:
        static_triples = np.array(_read_triplets_as_list("../data/" + args.dataset + "/e-w-graph.txt", {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes 
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().cuda(args.gpu) \
            if use_cuda else torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None
        
    if args.add_static_graph:
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)
    # create stat
    model_teacher = RecurrentRGCN(args.decoder,
                          args.encoder,
                        num_nodes,
                        num_rels,
                        num_static_rels,
                        num_words,
                        args.n_hidden_teacher,
                        args.opn,
                        sequence_len=args.train_history_len,
                        num_bases=args.n_bases,
                        num_basis=args.n_basis,
                        num_hidden_layers=args.n_layers_teacher,
                        dropout=args.dropout,
                        self_loop=args.self_loop,
                        skip_connect=args.skip_connect,
                        layer_norm=args.layer_norm,
                        input_dropout=args.input_dropout,
                        hidden_dropout=args.hidden_dropout,
                        feat_dropout=args.feat_dropout,
                        aggregation=args.aggregation,
                        weight=args.weight,
                        discount=args.discount,
                        angle=args.angle,
                        use_static=args.add_static_graph,
                        entity_prediction=args.entity_prediction,
                        relation_prediction=args.relation_prediction,
                        use_cuda=use_cuda,
                        gpu = args.gpu,
                        analysis=args.run_analysis)
    model_student = RecurrentRGCN(args.decoder,
                          args.encoder,
                        num_nodes,
                        num_rels,
                        num_static_rels,
                        num_words,
                        args.n_hidden_student,
                        args.opn,
                        sequence_len=args.train_history_len,
                        num_bases=args.n_bases,
                        num_basis=args.n_basis,
                        num_hidden_layers=args.n_layers_student,
                        dropout=args.dropout,
                        self_loop=args.self_loop,
                        skip_connect=args.skip_connect,
                        layer_norm=args.layer_norm,
                        input_dropout=args.input_dropout,
                        hidden_dropout=args.hidden_dropout,
                        feat_dropout=args.feat_dropout,
                        aggregation=args.aggregation,
                        weight=args.weight,
                        discount=args.discount,
                        angle=args.angle,
                        use_static=args.add_static_graph,
                        entity_prediction=args.entity_prediction,
                        relation_prediction=args.relation_prediction,
                        use_cuda=use_cuda,
                        gpu = args.gpu,
                        analysis=args.run_analysis)
    quality = quality_filter(1).cuda()
    value_entity = value_filter(num_nodes).cuda()
    value_relation = value_filter(num_rels*2).cuda()
    if use_cuda and args.role == 'teacher':
        torch.cuda.set_device(args.gpu)
        model = model_teacher
        model.cuda()
        model_name = "{}-{}-{}-ly{}-dilate{}-his{}-weight:{}-discount:{}-angle:{}-dp{}|{}|{}|{}|{}|{}|{}-gpu{}"\
        .format(args.dataset, args.encoder, args.decoder, args.n_layers_teacher, args.dilate_len, args.train_history_len, args.weight, args.discount, args.angle,
                args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout, args.role, args.situation, args.stage, args.gpu)
        model_state_file = '../models/' + model_name
        epochs = args.n_epochs_teacher
    elif use_cuda and args.role == 'student':
        torch.cuda.set_device(args.gpu)
        model = model_student
        model.cuda()
        model_name = "{}-{}-{}-ly{}-dilate{}-his{}-weight:{}-discount:{}-angle:{}-dp{}|{}|{}|{}|{}|{}|{}-gpu{}"\
            .format(args.dataset, args.encoder, args.decoder, args.n_layers_student, args.dilate_len, args.train_history_len, args.weight, args.discount, args.angle,
                    args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout, args.role, args.situation, args.stage, args.gpu)
        model_state_file = '../models/' + model_name
        epochs = args.n_epochs_student
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    if not args.test:
        print("----------------------------------------start training----------------------------------------\n")
            # optimizer
        if args.situation == 'pretrain':
            print("--------模型总的参数量---------")
            print(sum(p.numel() for p in model.parameters()))  # 打印模型参数量

            print("--------模型训练的参数量---------")
            print(sum(p.numel() for p in model.parameters() if p.requires_grad))  # 打印模型参数量
            #打印模型名称与shape
            # for name,parameters in model.named_parameters():
            #     print(name,':',parameters.size())
            train_function(epochs, model, train_list, num_nodes, num_rels, use_cuda, static_graph, optimizer, model_name, valid_list, all_ans_list_valid, all_ans_list_r_valid, model_state_file)
        
        elif args.situation == 'distill' and args.stage == 'stage1':
            model_state_file_teacher = '../models/' + "{}-{}-{}-ly{}-dilate{}-his{}-weight:{}-discount:{}-angle:{}-dp{}|{}|{}|{}|{}|{}|{}-gpu{}"\
        .format(args.dataset, args.encoder, args.decoder, args.n_layers_teacher, args.dilate_len, args.train_history_len, args.weight, args.discount, args.angle,
                args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout, 'teacher', 'pretrain', 'stage1', args.gpu)
            checkpoint = torch.load(model_state_file_teacher, map_location=torch.device(args.gpu))    
            # checkpoint = torch.load(model_state_file_teacher, map_location=torch.device('cpu'))
            model_tea = model_teacher.cuda()
            model_tea.load_state_dict(checkpoint['state_dict'])
            model_tea.eval()
            train_distill_stage1(args.n_epochs_stage1, quality, value_entity, value_relation, model, model_tea, train_list, num_nodes, num_rels, use_cuda, static_graph, optimizer, model_name, valid_list, all_ans_list_valid, all_ans_list_r_valid, model_state_file)
       
       
        elif args.situation == 'distill' and args.stage == 'stage2' and args.role == 'teacher':
            model_state_file_teacher = '../models/' + "{}-{}-{}-ly{}-dilate{}-his{}-weight:{}-discount:{}-angle:{}-dp{}|{}|{}|{}|{}|{}|{}-gpu{}"\
        .format(args.dataset, args.encoder, args.decoder, args.n_layers_student, args.dilate_len, args.train_history_len, args.weight, args.discount, args.angle,
                args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout, 'student', 'distill', 'stage1', args.gpu)
            checkpoint = torch.load(model_state_file_teacher, map_location=torch.device(args.gpu))
            model_tea = model_student.cuda()
            model_tea.load_state_dict(checkpoint['state_dict'])
            model_tea.eval()
            train_distill_stage2(epochs, value_entity, value_relation, model, model_tea, train_list, num_nodes, num_rels, use_cuda, static_graph, optimizer, model_name, valid_list, all_ans_list_valid, all_ans_list_r_valid, model_state_file)
        
        elif args.situation == 'distill' and args.stage == 'stage2' and args.role == 'student':
            model_state_file_teacher = '../models/' + "{}-{}-{}-ly{}-dilate{}-his{}-weight:{}-discount:{}-angle:{}-dp{}|{}|{}|{}|{}|{}|{}-gpu{}"\
        .format(args.dataset, args.encoder, args.decoder, args.n_layers_teacher, args.dilate_len, args.train_history_len, args.weight, args.discount, args.angle,
                args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout, 'teacher', 'distill', 'stage2', args.gpu)
            checkpoint = torch.load(model_state_file_teacher, map_location=torch.device(args.gpu))
            model_tea = model_teacher.cuda()
            model_tea.load_state_dict(checkpoint['state_dict'])
            model_tea.eval()
            train_distill_stage1(args.n_epochs_stage2, quality, value_entity, value_relation, model, model_tea, train_list, num_nodes, num_rels, use_cuda, static_graph, optimizer, model_name, valid_list, all_ans_list_valid, all_ans_list_r_valid, model_state_file)
   
   
    elif args.test and os.path.exists(model_state_file):
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
                                                            train_list+valid_list, 
                                                            test_list, 
                                                            num_rels, 
                                                            num_nodes, 
                                                            use_cuda, 
                                                            all_ans_list_test, 
                                                            all_ans_list_r_test, 
                                                            model_state_file, 
                                                            static_graph, 
                                                            "test")
    elif args.test and not os.path.exists(model_state_file):
        print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(model_state_file))
               
    mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
                                                            train_list+valid_list,
                                                            test_list, 
                                                            num_rels, 
                                                            num_nodes, 
                                                            use_cuda, 
                                                            all_ans_list_test, 
                                                            all_ans_list_r_test, 
                                                            model_state_file, 
                                                            static_graph, 
                                                            mode="test")
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='REGCN')

    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--role", type=str, required=True,
                        help="role to use")
    parser.add_argument("--situation", type=str, required=True,
                        help="situation to use")
    parser.add_argument("--stage", type=str, required=True,
                        help="stage to use")
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")
    parser.add_argument("--multi-step", action='store_true', default=False,
                        help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=10,
                        help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-static-graph",  action='store_true', default=False,
                        help="use the info of static graph")
    parser.add_argument("--add-rel-word", action='store_true', default=False,
                        help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")

    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of static constraint")
    parser.add_argument("--task-weight", type=float, default=0.7,
                        help="weight of entity prediction task")
    parser.add_argument("--discount", type=float, default=1,
                        help="discount of weight of static constraint")
    parser.add_argument("--angle", type=int, default=10,
                        help="evolution speed")

    parser.add_argument("--encoder", type=str, default="uvrgcn",
                        help="method of encoder")
    parser.add_argument("--aggregation", type=str, default="none",
                        help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n_hidden_teacher", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--n_hidden_student", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n_layers_teacher", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--n_layers_student", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=False,
                        help="add entity prediction loss")
    parser.add_argument("--split_by_relation", action='store_true', default=False,
                        help="do relation prediction")

    # configuration for stat training
    parser.add_argument("--n-epochs_teacher", type=int, default=10,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--n-epochs_student", type=int, default=4,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--n-epochs_stage1", type=int, default=8,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--n-epochs_stage2", type=int, default=10,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=1,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=10,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=20,
                        help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")

    # configuration for optimal parameters
    parser.add_argument("--grid-search", action='store_true', default=False,
                        help="perform grid search for best configuration")
    parser.add_argument("-tune", "--tune", type=str, default="n_hidden,n_layers,dropout,n_bases",
                        help="stat to use")
    parser.add_argument("--num-k", type=int, default=500,
                        help="number of triples generated")


    args = parser.parse_args()
    print(args)
    if args.grid_search:
        out_log = '{}.{}.gs'.format(args.dataset, args.encoder+"-"+args.decoder)
        o_f = open(out_log, 'w')
        print("** Grid Search **")
        o_f.write("** Grid Search **\n")
        hyperparameters = args.tune.split(',')

        if args.tune == '' or len(hyperparameters) < 1:
            print("No hyperparameter specified.")
            sys.exit(0)
        grid = hp_range[hyperparameters[0]]
        for hp in hyperparameters[1:]:
            grid = itertools.product(grid, hp_range[hp])
        hits_at_1s = {}
        hits_at_10s = {}
        mrrs = {}
        grid = list(grid)
        print('* {} hyperparameter combinations to try'.format(len(grid)))
        o_f.write('* {} hyperparameter combinations to try\n'.format(len(grid)))
        o_f.close()

        for i, grid_entry in enumerate(list(grid)):

            o_f = open(out_log, 'a')

            if not (type(grid_entry) is list or type(grid_entry) is list):
                grid_entry = [grid_entry]
            grid_entry = utils.flatten(grid_entry)
            print('* Hyperparameter Set {}:'.format(i))
            o_f.write('* Hyperparameter Set {}:\n'.format(i))
            signature = ''
            print(grid_entry)
            o_f.write("\t".join([str(_) for _ in grid_entry]) + "\n")
            # def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
            mrr, hits, ranks = run_experiment(args, grid_entry[0], grid_entry[1], grid_entry[2], grid_entry[3])
            print("MRR (raw): {:.6f}".format(mrr))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr))
            for hit in hits:
                avg_count = torch.mean((ranks <= hit).float())
                print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, avg_count.item()))
    # single run
    else:
        run_experiment(args)
    sys.exit()



