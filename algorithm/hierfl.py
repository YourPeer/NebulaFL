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
        for r in range(self.args.rounds):
            lock.acquire()
            self.T+=1
            print("======================== Round: %d ========================"%(r))
            self.async_aggregation()
            test_acc, test_loss = self.test_model()
            print("Test_loss: %.3f | Test_accuracy: %.2f" % (test_loss, test_acc))
            print("===========================================================")
            lock.release()

    def async_aggregation(self):
        info, indices = self.pack()
        leader_info = torch.zeros_like(info)
        dist.recv(leader_info)
        leader_params, id, tao = self.unpack(leader_info, indices)
        global_params = [p.data.view(-1, 1) for p in self.model.parameters()]
        alpha_t = self.s(self.T - tao)
        print(alpha_t)
        global_params=[gp*(1-alpha_t)+lp*alpha_t for lp,gp in zip(leader_params,global_params)]
        self.set_model(global_params)
        info, indices = self.pack()
        dist.send(info, id)

    def pack(self):
        info = [torch.flatten(p.data) for p in self.model.parameters()]
        extra_info = torch.tensor([0,self.T])
        info.append(extra_info)
        indices = []
        s = 0
        for i in info:
            size = i.size()[0]
            indices.append((s, s + size))
            s += size
        info = torch.cat(info).view((-1, 1))
        return info, indices

    def unpack(self, info, indices):
        l = [info[s:e] for (s, e) in indices]
        model_param = l[:-1]
        extra_info = l[-1].view(-1).tolist()
        id = int(extra_info[0])
        tao=int(extra_info[1])
        return model_param, id, tao

    def s(self, delta_tau):
        tiers = self.args.algorithms[self.args.algorithm][1]["tiers"]
        flag = self.args.algorithms[self.args.algorithm][1]["flag"]
        alpha = self.args.algorithms[self.args.algorithm][1]["alpha"]
        if flag=="constant":
            return 1 / tiers
        else:
            return 1/tiers if delta_tau <= tiers else 1.0 / (tiers*((delta_tau-tiers +1)**alpha))


class Client(BasicClient):

    def run(self):
        self.init_process()
        self.model.train()
        self.tao=0
        for round in range(self.args.rounds):
             self.local_train()
             self.communicate_with_server()



    def communicate_with_server(self):
        tier_info=self.get_tiers()
        if self.client_id in tier_info.keys():
            leader_node=tier_info[self.client_id]
            info, indices=self.pack()
            send_info(info,leader_node)
            recv_info(info,leader_node)
            model_param, id = self.unpack(info, indices)
            self.set_model(model_param)  # set all subnodes model
        else:
            sub_node_list = [key for key, value in tier_info.items() if value == self.client_id]
            self.aggregation(sub_node_list)


    def aggregation(self,sub_node_list): # choose a leader node for sync aggregation
        sub_node_list.append(self.client_id)
        info, indices = self.pack()
        clients_num = len(sub_node_list)
        info_list = [torch.zeros_like(info) for _ in range(clients_num-1)]
        self.set_ratio(sub_node_list)
        global_params_list = [p.data.view(-1, 1)*self.ratio[-1] for p in self.model.parameters()]
        for i, k in enumerate(sub_node_list[:-1]):
            recv_info(info_list[i], k)
            local_params_list, id = self.unpack(info_list[i], indices)
            global_params_list = [lp * self.ratio[i] + gp for lp, gp in zip(local_params_list, global_params_list)]
        self.set_model(global_params_list)
        info, indices = self.pack() # to server
        try:
            send_info(info, self.server_id)
        except:
            # print("leader_node %d terminated." % (self.client_id))
            sys.exit()
        recv_info(info, self.server_id) # recv from server
        global_params_list, id = self.unpack(info, indices)
        self.set_model(global_params_list)  # set leader node model
        info, indices = self.pack() # to all subnodes
        for i in sub_node_list[:-1]:
            send_info(info, i)

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

    def pack(self):
        info = [torch.flatten(p.data) for p in self.model.parameters()]
        extra_info = torch.tensor([self.client_id,self.tao])
        info.append(extra_info)
        indices = []
        s = 0
        for i in info:
            size = i.size()[0]
            indices.append((s, s + size))
            s += size
        info = torch.cat(info).view((-1, 1))
        return info, indices

    def unpack(self,info,indices):
        l = [info[s:e] for (s, e) in indices]
        model_param=l[:-1]
        extra_info = l[-1].view(-1).tolist()
        id = int(extra_info[0])
        self.tao=int(extra_info[1])
        return model_param, id

