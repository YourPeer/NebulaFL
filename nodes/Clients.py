import os
import random
import torch
import torch.distributed as dist
import numpy as np
from tools import *
from .comm_op import *
class Client(object):
    def __init__(self, client_id, args):
        super().__init__()
        self.server_id=args.clients-1
        self.client_id=client_id
        self.args = args

    def init_process(self):
        torch.manual_seed(2022)
        torch.cuda.manual_seed(2022)
        torch.cuda.manual_seed_all(2022)
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_device(self.client_id % self.args.gpu_nums)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12340'
        dist.init_process_group("gloo", rank=self.client_id, world_size=self.args.clients)
        self.setup_seed(2222)
        torch.cuda.set_device(self.client_id % self.args.gpu_nums)

        self.train_loader=self.args.distributer["local"][self.client_id]["train"]
        self.model=create_models(self.args.modelname,self.args.dataset)

        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.args.lr,
                                         momentum=self.args.momentum)

    def get_sample_clients(self):
        sample_clients=torch.tensor([0]*int(self.args.clients*self.args.sample_ratio))
        broadcast(sample_clients,self.server_id)
        return sample_clients

    def run(self):
        self.init_process()
        self.model.train()
        for round in range(self.args.rounds):
            sample_clients=self.get_sample_clients()
            if self.client_id in sample_clients:
                self.local_train()
                self.communicate_with_server()

    def local_train(self):
        self.model.train()
        self.model.cuda()
        for _ in range(self.args.epochs):
            for data, targets in self.train_loader:
                self.optimizer.zero_grad()
                # forward pass
                data, targets = data.cuda(), targets.cuda()
                output = self.model(data)
                loss = self.criterion(output, targets)
                # backward pass
                loss.backward()
                self.optimizer.step()
        self.model.cpu()

    def communicate_with_server(self):
        info, indices=self.pack()
        send_info(info,self.server_id)
        recv_info(info,self.server_id)
        model_param, extra_info = self.unpack(info,indices)
        self.set_model(model_param)

    def pack(self):
        info = [torch.flatten(p.data) for p in self.model.parameters()]
        extra_info = torch.tensor([self.client_id])
        info.append(extra_info)
        indices = []
        s = 0
        for i in info:
            size=i.size()[0]
            indices.append((s, s + size))
            s += size
        info = torch.cat(info).view((-1, 1))
        return info, indices

    def unpack(self,info,indices):
        l = [info[s:e] for (s, e) in indices]
        model_param=l[:-1]
        extra_info=l[-1]
        return model_param,extra_info

    def set_model(self,model_param):
        for i,p in enumerate(self.model.parameters()):
            p.data= model_param[i].view(*p.shape)

    def setup_seed(self,seed):
        r"""
        Fix all the random seed used in numpy, torch and random module

        Args:
            seed (int): the random seed
        """
        if seed < 0:
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            seed = -seed
        random.seed(1 + seed)
        np.random.seed(21 + seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(12 + seed)
        torch.cuda.manual_seed_all(123 + seed)
