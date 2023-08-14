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
            id, self.train_loss=self.async_aggregation()
            test_acc, test_loss = self.test_model()
            self.args.logger.info("Test_loss: %.3f | Test_accuracy: %.3f | Train_loss: %.3f | Client: %d" % (test_loss, test_acc, self.train_loss, id))
            self.args.logger.info("===========================================================")
            lock.release()

    def async_aggregation(self):
        info, indices = self.pack()
        leader_info = torch.zeros_like(info)
        dist.recv(leader_info)
        leader_params, extra_info = self.unpack(leader_info, indices)
        id,tao,train_loss=self.get_extra_info(extra_info)
        global_params = [p.data.view(-1, 1) for p in self.model.parameters()]
        alpha_t = self.s(self.T - tao)
        global_params=[gp*(1-alpha_t)+lp*alpha_t for lp,gp in zip(leader_params,global_params)]
        self.set_model(global_params)
        info, indices = self.pack()
        dist.send(info, id)
        return  id, train_loss

    def add_extra_info(self):
        extra_info_dict={
            "client_id":[self.server_id],
            "tao": [self.T],
            "train_loss":[self.train_loss]
        }
        return extra_info_dict

    def get_extra_info(self,extra_info):
        id=int(extra_info[0].item())
        tao=int(extra_info[1].item())
        train_loss=extra_info[2].item()
        return id,tao,train_loss

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
        self.local_train_loss=0.0
        for round in range(self.args.rounds):
             self.local_train()
             self.local_train_loss = self.test_model()
             try:
                self.communicate_with_server()
             except:
             # print("leader_node %d terminated." % (self.client_id))
                sys.exit()

    def add_extra_info(self):
        extra_info_dict={
            "client_id":[self.client_id],
            "tao": [self.tao],
            "train_loss":[self.local_train_loss]
        }
        return extra_info_dict

