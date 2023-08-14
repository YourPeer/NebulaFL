import copy

from nodes import BasicServer
from nodes import BasicClient
from nodes import send_info,recv_info
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import sys
import time
class Server(BasicServer):
    def run(self):
        lock = torch.multiprocessing.Lock()
        self.init_process()
        self.T=0
        self.train_loss=0.0
        for r in range(self.args.rounds):
            lock.acquire()
            self.T+=1
            self.args.logger.info("======================== Round: %d ========================"%(r))
            self.train_loss, id = self.async_aggregation()
            test_acc, test_loss = self.test_model()
            self.args.logger.info("Test_loss: %.3f | Test_accuracy: %.2f | Tier_train_loss: %.3f | Edge_server_id (Tier leader): %d" % (test_loss, test_acc, self.train_loss, id))
            # self.args.logger.info(
            #     "Tier_train_loss_list: %s " % (str(tier_train_loss_list)))
            self.args.logger.info("===========================================================")
            lock.release()

    def async_aggregation(self):
        info, indices = self.pack()
        leader_info = torch.zeros_like(info)
        dist.recv(leader_info)
        leader_params, extra_info = self.unpack(leader_info, indices)
        id, tao, train_loss=self.get_extra_info(extra_info)
        global_params = [p.data.view(-1, 1) for p in self.model.parameters()]
        alpha_t = self.s(self.T - tao)
        global_params=[gp*(1-alpha_t)+lp*alpha_t for lp,gp in zip(leader_params,global_params)]
        self.set_model(global_params)
        info, indices = self.pack()
        dist.send(info, id)
        return train_loss, id

    def add_extra_info(self):
        extra_info_dict = {
            "client_id": [self.server_id],
            "tao": [self.T],
            "train_loss": [self.train_loss]
        }
        return extra_info_dict

    def get_extra_info(self, extra_info):
        id = int(extra_info[0].item())
        tao = int(extra_info[1].item())
        train_loss = extra_info[2].item()
        return id, tao, train_loss

    def s(self, delta_tau):
        if self.flag=="constant":
            return self.alpha
        else:
            return 1/self.tiers if delta_tau <= self.tiers else 1.0 / (self.tiers*((delta_tau-self.tiers +1)**self.alpha))


class Client(BasicClient):

    def run(self):
        self.init_process()
        self.model.train()
        self.tao=0
        self.train_loss=0.0
        tier_info = self.get_tiers()
        for round in range(self.args.rounds):
             self.local_train()
             self.train_loss = self.test_model()

             try:
                self.communicate_with_server(tier_info)
             except Exception as e:
                 print(type(e).__name__ + ": " + str(e))
                 sys.exit()

    def communicate_with_server(self,tier_info):
        if self.client_id in tier_info.values(): # Edge server
            sub_node_list = [key for key, value in tier_info.items() if value == self.client_id]
            self.aggregation(sub_node_list)
        else:
            edge_server_id=tier_info[self.client_id]
            info, indices=self.pack()
            send_info(info,edge_server_id)
            recv_info(info,edge_server_id)
            model_param, extro_info = self.unpack(info,indices)
            self.set_model(model_param)

    def aggregation(self,sub_node_list): # choose a leader node for sync aggregation
        sub_node_list.append(self.client_id) # append edge_server
        info, indices = self.pack()
        clients_num = len(sub_node_list)
        info_list = [torch.zeros_like(info) for _ in range(clients_num-1)]
        self.set_ratio(sub_node_list)
        global_params_list = [p.data.view(-1, 1)*self.ratio[-1] for p in self.model.parameters()]
        train_loss_list=[self.train_loss]
        for i, k in enumerate(sub_node_list[:-1]):
            recv_info(info_list[i], k)
            local_params_list, extra_info  = self.unpack(info_list[i], indices)
            id, tao, train_loss = self.get_extra_info(extra_info)
            train_loss_list.append(train_loss)
            global_params_list = [lp * self.ratio[i] + gp for lp, gp in zip(local_params_list, global_params_list)]

        self.set_model(global_params_list)
        self.train_loss=np.mean(train_loss_list)  # float64 to float32
        info, indices = self.pack() # to server
        send_info(info, self.server_id)
        dist.recv(info) # recv from server
        global_params_list, extro_info= self.unpack(info, indices)
        self.set_model(global_params_list)  # set leader node model
        info, indices = self.pack() # to all subnodes
        for i in sub_node_list[:-1]:
            send_info(info, i)
        

    def add_extra_info(self):
        extra_info_dict={
            "client_id":[self.client_id],
            "tao": [self.tao],
            "train_loss":[self.train_loss]
        }
        return extra_info_dict

    def get_extra_info(self,extra_info):
        id=int(extra_info[0].item())
        tao=int(extra_info[1].item())
        train_loss=extra_info[2].item()
        return id,tao,train_loss

    def set_ratio(self, sub_node_list):
        clients_num = len(sub_node_list)
        if self.args.aggregate_methods == "uniform":
            self.ratio=[1/clients_num]*clients_num
        elif self.args.aggregate_methods == "weight":
            ratio = self.args.data_ratio[sub_node_list]
            self.ratio = [r/np.sum(ratio) for r in ratio]

    def get_tiers(self):
        clients_num=self.args.clients-1
        tiers=self.args.algorithms[self.args.algorithm][1]["tiers"]
        clients_index = np.array(range(clients_num)).reshape(tiers, -1)
        tier_info = {}
        for k in clients_index:
            for i in k[1:]:
                tier_info[i] = k[0]
        return tier_info





