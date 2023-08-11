import numpy as np


def get_tiers():
    clients_num = 20
    tiers = 5
    clients_index = np.array(range(clients_num)).reshape(tiers, -1)
    tier_info = {}
    for k in clients_index:
        tier_info[k[0]] = k
    return tier_info

print(get_tiers())