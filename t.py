import numpy as np

clients=range(20)
rs = np.random.normal(1.0, 1.0, len(clients))
rs = rs.clip(0.01, 2)
_my_working_amount = {cid:max(int(r*5),1) for  cid,r in zip(clients, rs)}

print(_my_working_amount)