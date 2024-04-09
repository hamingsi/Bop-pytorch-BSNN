import os
import time
import argparse
import data_loaders
from torch.utils.data import DataLoader
from models.resnet_models import *
from models.VGG_models import *
from functions import proposedLoss, seed_all, get_logger, Log_UP, res19KT, res18KT
from Bop import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('-----------Bop binary model------------')


parser = argparse.ArgumentParser(description='PyTorch Bop BSNN train')
parser.add_argument('-j',
                    '--workers',
                    default=16,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--seed',
                    default=3407,
                    type=int,
                    metavar='S',
                    help='random seed (default: 3407)')
parser.add_argument('--epochs',
                    default=300,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--b',
                    default=64,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-lr',
                    '--learning_rate',
                    default=1e-1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('-T',
                    '--time',
                    default=2,
                    type=int,
                    metavar='N',
                    help='snn simulation time (default: 2)')
parser.add_argument('--ourLoss',
                    default=False,
                    type=bool,
                    metavar='N',
                    help='if use the proposed loss function (default: True)')
parser.add_argument('--lamb_f',
                    default=5e-3,
                    type=float,
                    metavar='N')
parser.add_argument('--dataset',
                    default='CIFAR10',
                    type=str,
                    help='dataset name',
                    choices=['CIFAR10', 'CIFAR100', 'ImageNet'])
parser.add_argument('--arch',
                    default='res19',
                    type=str,
                    help='model',
                    choices=['res19','res18'])
args = parser.parse_args()


def train(model, device, train_loader, criterion, optimizer, epoch, args):
    running_loss = 0

    model.train()

    total = 0
    correct = 0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)
        outputs, fr = model(images)
        mean_out = outputs.mean(0)

        if args.ourLoss:
            loss = proposedLoss(outputs, fr, labels, criterion, args.lamb_f)
        else:
            loss = criterion(mean_out,labels)
        running_loss += loss.item()
        loss.mean().backward()
        
        optimizer.step()
        
        total += float(labels.size(0))
        _, predicted = mean_out.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total

@torch.no_grad()
def test(model, test_loader, device):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs, _ = model(inputs)
        mean_out = outputs.mean(0)
        _, predicted = mean_out.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        if batch_idx % 100 == 0:
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    final_acc = 100 * correct / total
    return final_acc

if __name__ == '__main__':
    seed_all(args.seed)

    # build dataset
    if args.dataset == 'CIFAR10':
        trainset, testset = data_loaders.build_cifar(cutout=True,use_cifar10=True,download=True)
    elif args.dataset == 'CIFAR100':
        trainset, testset = data_loaders.build_cifar(cutout=True,use_cifar10=False,download=False)
    elif args.dataset == 'ImageNet':
        trainset, testset = data_loaders.build_imagenet()
    elif args.dataset == 'DVSCIFAR10':
        trainset, testset = data_loaders.build_dvscifar()

    train_loader = DataLoader(trainset, batch_size=args.b, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=args.b, shuffle=False, num_workers=args.workers, pin_memory=True)

    # build model
    # if args.arch == 'res19':
    #     model = resnet19()
    # elif args.arch == 'res18':
    #     model = resnet18()
    model = VGG11_BSNN()
    parallel_model = torch.nn.DataParallel(model)
    parallel_model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    parameters = split_weights(model)
    # optimizer = torch.optim.SGD(params=parameters, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    optimizer = Bop(parameters,lr=args.lr)
    # optimizer = torch.optim.Adam(params=parameters, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    best_acc = 0
    best_epoch = 0
    
    logger = get_logger('Bop_changeThres1e-6gam1e-3.log')
    logger.info('start training!')
    
    for epoch in range(args.epochs):
        loss, acc = train(parallel_model, device, train_loader, criterion, optimizer, epoch, args)
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch , args.epochs, loss, acc ))
        scheduler.step()
        facc = test(parallel_model, test_loader, device)
        logger.info('Epoch:[{}/{}]\t Test acc={:.3f}'.format(epoch , args.epochs, facc ))

        if best_acc < facc:
            best_acc = facc
            best_epoch = epoch + 1
            torch.save(parallel_model.module.state_dict(), 'res19_1w2u.pth')
        logger.info('Best Test acc={:.3f}'.format(best_acc ))
        print('\n')
