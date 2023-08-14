import torch
import torch.distributed as dist

def send_info(info,dst):
    dist.send(info,dst)

def recv_info(info,src):
    dist.recv(info,src)

def broadcast(info,src):
    dist.broadcast(info, src=src)