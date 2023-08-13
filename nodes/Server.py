import os
import random
import torch
import torch.distributed as dist
import numpy as np
from tools import *
from .comm_op import *

class Server(object):
    def __init__(self, args):
        super().__init__()
        self.server_id=args.clients-1
        self.args = args

    def init_process(self):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12340'
        dist.init_process_group("gloo", rank=self.server_id, world_size=self.args.clients)
        self.setup_seed(self.args.seed)
        self.test_loader=self.args.distributer["global"]["test"]
        self.model=create_models(self.args.modelname,self.args.dataset)
        self.init_algo_para(self.args.algorithms[self.args.algorithm][1])
        print("Server Init Network Success!")

    def _client_sampling(self, round_idx):
        """Sample clients by given sampling ratio"""
        # make sure for same client sampling for fair comparison
        np.random.seed(round_idx+1234)
        clients_per_round = max(int((self.args.clients-1) * self.args.sample_ratio), 1)
        sampled_clients = np.random.choice(
            range(0,self.args.clients-1), clients_per_round, replace=False
        )
        info = torch.tensor(sampled_clients)
        print("Sample client list: %s"%str(info.numpy().tolist()))
        dist.broadcast(info,self.server_id)
        return info

    def run(self):
        self.init_process()

        for r in range(self.args.rounds):
            print("======================== Round: %d ========================"%(r))
            self.sample_clients=self._client_sampling(r)
            self.aggregation()
            test_acc,test_loss=self.test_model()
            print("Test_loss: %.3f | Test_accuracy: %.2f"%(test_loss,test_acc))
            print("===========================================================")


    def set_ratio(self):
        clients_num = len(self.sample_clients)
        if self.args.aggregate_methods == "uniform":
            self.ratio=[1/clients_num]*clients_num
        elif self.args.aggregate_methods == "weight":
            ratio = self.args.data_ratio[self.sample_clients]
            self.ratio = [r/np.sum(ratio) for r in ratio]

    def aggregation(self):
        info, indices=self.pack()
        clients_num=len(self.sample_clients)
        info_list=[torch.zeros_like(info) for _ in range(clients_num)]
        global_params_list = [torch.zeros_like(p.data.view(-1,1)) for p in self.model.parameters()]
        self.set_ratio()
        for i,k in enumerate(self.sample_clients):
            recv_info(info_list[i],k)
            local_params_list, id=self.unpack(info_list[i], indices)
            global_params_list=[lp*self.ratio[i]+gp for lp,gp in zip(local_params_list,global_params_list)]
        self.set_model(global_params_list)
        info, indices=self.pack()
        for i in self.sample_clients:
            send_info(info,i)

    def pack(self):
        info = [torch.flatten(p.data) for p in self.model.parameters()]
        extra_info = torch.tensor([0])
        info.append(extra_info)
        indices=[]
        s=0
        for i in info:
            size=i.size()[0]
            indices.append((s,s+size))
            s+=size
        info=torch.cat(info).view((-1,1))
        return info, indices

    def unpack(self,info,indices):
        l = [info[s:e] for (s, e) in indices]
        model_param=l[:-1]
        extra_info = l[-1]
        return model_param, extra_info

    def set_model(self,model_param):
        for i,p in enumerate(self.model.parameters()):
            p.data= model_param[i].view(*p.shape)


    def test_model(self):
        self.model.cuda()
        self.model.eval()
        test_loss = 0.0
        correct = 0.0
        total = 0.0
        criterion = torch.nn.CrossEntropyLoss().cuda()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        self.model.train()
        self.model.cpu()
        return 100. * correct / total, test_loss / (batch_idx + 1)

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



