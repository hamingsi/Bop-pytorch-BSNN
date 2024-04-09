import torch
import torch.nn as nn
import random
import os
import numpy as np
import logging
import math

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def proposedLoss(outputs, fire_rate, labels, criterion, lamb):
    T = outputs.size(0)
    Loss_es, Loss_f = 0, 0
    for t in range(T):
        Loss_es += criterion(outputs[t, ...], labels)
    Loss_es = Loss_es / T
    if lamb != 0:
        Loss_f = (sum([Entro(ele) for ele in fire_rate[1:-1]])) / len(fire_rate[1:-1])  # del the first and the last layer
    return Loss_es + lamb * Loss_f # L_Total

def Entro(rate):
    return (rate - 0.5) ** 2

def Log_UP(K_min, K_max, Epochs, epoch):
    Kmin, Kmax = math.log(K_min) / math.log(10), math.log(K_max) / math.log(10)
    return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / Epochs * epoch)]).float().cuda()

def res18KT(model, k, t):
    for i in range(2):  # iter of the basicblock
        model.layer1[i].conv1.k = k
        model.layer1[i].conv2.k = k
        model.layer1[i].conv1.t = t
        model.layer1[i].conv2.t = t

        model.layer2[i].conv1.k = k
        model.layer2[i].conv2.k = k
        model.layer2[i].conv1.t = t
        model.layer2[i].conv2.t = t

        model.layer3[i].conv1.k = k
        model.layer3[i].conv2.k = k
        model.layer3[i].conv1.t = t
        model.layer3[i].conv2.t = t

        model.layer4[i].conv1.k = k
        model.layer4[i].conv2.k = k
        model.layer4[i].conv1.t = t
        model.layer4[i].conv2.t = t

    return model

def res19KT(model, k, t):

    for i in range(3):  # iter of the basicblock
        model.layer1[i].conv1.k = k
        model.layer1[i].conv2.k = k
        model.layer1[i].conv1.t = t
        model.layer1[i].conv2.t = t

        model.layer2[i].conv1.k = k
        model.layer2[i].conv2.k = k
        model.layer2[i].conv1.t = t
        model.layer2[i].conv2.t = t

        # model.layer1[i].conv1.module.k = k
        # model.layer1[i].conv2.module.k = k
        # model.layer1[i].conv1.module.t = t
        # model.layer1[i].conv2.module.t = t
        #
        # model.layer2[i].conv1.module.k = k
        # model.layer2[i].conv2.module.k = k
        # model.layer2[i].conv1.module.t = t
        # model.layer2[i].conv2.module.t = t

        if i < 2:
            model.layer3[i].conv1.k = k
            model.layer3[i].conv2.k = k
            model.layer3[i].conv1.t = t
            model.layer3[i].conv2.t = t
            # model.layer3[i].conv1.module.k = k
            # model.layer3[i].conv2.module.k = k
            # model.layer3[i].conv1.module.t = t
            # model.layer3[i].conv2.module.t = t

    return model