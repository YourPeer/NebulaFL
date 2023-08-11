from nodes import BasicServer
from nodes import BasicClient
from nodes import send_info,recv_info
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import sys
class Server(BasicServer):
    def run(self):
        lock = torch.multiprocessing.Lock()
        self.init_process()
        for r in range(self.args.rounds):
            lock.acquire()
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
        leader_params, id = self.unpack(leader_info, indices)
        global_params = [p.data.view(-1, 1) for p in self.model.parameters()]
        global_params=[gp*0.7+lp*0.3 for lp,gp in zip(leader_params,global_params)]
        self.set_model(global_params)
        info, indices = self.pack()
        dist.send(info, id)

class Client(BasicClient):

    def run(self):
        self.init_process()
        self.model.train()
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
            model_param, extra_info = self.unpack(info, indices)
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
        recv_info(info,self.server_id)
        info, indices = self.pack() # to all subnodes
        self.set_model(global_params_list) # set leader node model
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

