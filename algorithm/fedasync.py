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
        clients=self.args.clients
        if self.flag=="constant":
            return 1 / clients
        else:
            return 1/clients if delta_tau <= clients else 1.0 / (clients*((delta_tau-clients +1)**self.alpha))


class Client(BasicClient):

    def run(self):
        self.init_process()
        self.model.train()
        self.tao=0
        for round in range(self.args.rounds):
             self.local_train()
             try:
                self.communicate_with_server()
             except:
             # print("leader_node %d terminated." % (self.client_id))
                sys.exit()

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

