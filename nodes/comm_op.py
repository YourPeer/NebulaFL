import torch
import torch.distributed as dist

def send_info(info,src):
    dist.send(info,src)

def recv_info(info,src):
    dist.recv(info,src)

def broadcast(info,src): # info=["foo", 12, {1: 2}]
    dist.broadcast(info, src=src)