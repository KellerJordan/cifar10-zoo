import os
import uuid
import torch
#from airbench_lite import train
#from airbench_medium import train
from __init__ import train94
out_dir = './logs'
for i in range(5000):
    net = train(i)
    sd = net.state_dict()
    k = str(uuid.uuid4())
    torch.save(sd, os.path.join(out_dir, k+'.pt'))

