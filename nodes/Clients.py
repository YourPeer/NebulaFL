import copy
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
        self.setup_seed(self.args.seed)
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
        self.current_steps=0
        self.num_steps=self.args.epochs*len(self.train_loader)
        self.init_algo_para(self.args.algorithms[self.args.algorithm][1])

    def get_sample_clients(self):
        sample_clients=torch.tensor([0]*int(self.args.clients*self.args.sample_ratio))
        broadcast(sample_clients,self.server_id)
        return sample_clients

    def run(self):
        self.init_process()
        self.model.train()

        for round in range(self.args.rounds):
            self.sample_clients=self.get_sample_clients()
            if self.client_id in self.sample_clients:
                self.local_train()
                self.local_train_loss = self.test_model()
                self.communicate_with_server()

    def local_train(self):
        self.model.train()
        self.model.cuda()
        for _ in range(self.num_steps):
            (data, targets) = self.get_batch_data()
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
        extra_info = torch.tensor([self.client_id, self.local_train_loss])
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


    def init_algo_para(self, algo_para: dict):
        self.algo_para = algo_para
        if len(self.algo_para) == 0: return
        # register the algorithm-dependent hyperparameters as the attributes of the server and all the clients
        for para_name, value in self.algo_para.items():
            self.__setattr__(para_name, value)
        return

    def get_batch_data(self):
        """
        Get the batch of training data
        Returns:
            a batch of data
        """
        try:
            batch_data = next(self.data_loader)
        except Exception as e:
            self.data_loader = iter(self.train_loader)
            batch_data = next(self.data_loader)
        # clear local_movielens_recommendation DataLoader when finishing local_movielens_recommendation training
        self.current_steps = (self.current_steps + 1) % self.num_steps
        if self.current_steps == 0:
            self.data_loader = None
            self._train_loader = None
        return batch_data

    def test_model(self):
        self.model.cuda()
        self.model.eval()
        train_loss = 0.0
        #correct = 0.0
        total = 0.0
        criterion = torch.nn.CrossEntropyLoss().cuda()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                #correct += predicted.eq(targets).sum().item()
        self.model.train()
        self.model.cpu()
        local_train_loss=train_loss / (batch_idx + 1)
        # correct=100. * correct / total
        return local_train_loss
