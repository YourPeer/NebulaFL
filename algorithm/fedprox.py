from nodes import BasicServer as Server
from nodes import BasicClient
import copy
import torch

class Client(BasicClient):
    def local_train(self):
        # global parameters
        src_model = copy.deepcopy(self.model)
        for p in src_model.parameters():
            p.requires_grad = False
        self.model.train()
        self.model.cuda()
        src_model.cuda()
        for _ in range(self.args.epochs):
            for data, targets in self.train_loader:
                self.optimizer.zero_grad()
                data, targets = data.cuda(), targets.cuda()
                output = self.model(data)
                loss = self.criterion(output, targets)
                loss_proximal = 0
                for pm, ps in zip(self.model.parameters(), src_model.parameters()):
                    loss_proximal += torch.sum(torch.pow(pm - ps, 2))
                loss = loss + 0.5 * self.args.algorithms[self.args.algorithm][1]["mu"] * loss_proximal
                loss.backward()
                self.optimizer.step()
        src_model.cpu()
        self.model.cpu()