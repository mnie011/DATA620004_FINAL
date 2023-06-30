import argparse
from config import cfg, cfg_from_yaml_file
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.tensorboard import SummaryWriter
from vae import VAEResNet18
from utils import progress_bar
from loss_utils import VAECriterion
from torch.autograd import Variable


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='../cfgs/ssup/vae-r18.yaml', help='specify the config for training')
    parser.add_argument('--model_name', type=str, default='VAE-R18')
    parser.add_argument('--train_batch_size', type=int, default=None, required=False, help='batch_size for training')
    parser.add_argument('--test_batch_size', type=int, default=None, required=False)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=150, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=None, required=False)
    parser.add_argument('--momentum', type=float, default=None, required=False)
    parser.add_argument('--weight_decay', type=float, default=None, required=False)
    parser.add_argument('--kld_weight', type=float, default=None, required=False)

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def build_criterion(kld_weight):
    return VAECriterion(kld_weight)


def main():
    args, cfg = parse_config()

    start_epoch = args.start_epoch
    epochs = args.epochs

    if args.train_batch_size is None:
        args.train_batch_size = cfg.TRAIN_BATCH_SIZE
        args.test_batch_size = cfg.TEST_BATCH_SIZE

    if args.lr is None:
        args.lr = cfg.LR
        args.momentum = cfg.MOMENTUM
        args.weight_decay = cfg.WEIGHT_DECAY
        args.kld_weight = cfg.KLD_WEIGHT

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    val_percent = 0.2
    trainset = torchvision.datasets.CIFAR100(
        root='../muse', train=True, download=True, transform=transform_train)
    classes = trainset.classes
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    print('==> Building model..')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.model_name == 'VAE-R18':
        net = VAEResNet18()
    else:
        raise NotImplementedError

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('../muse/checkpoints'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('../muse/checkpoints/{}.pth'.format(args.model_name))
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']

    criterion = build_criterion(args.kld_weight)
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    writer = SummaryWriter('./path/{}/log'.format(args.model_name))

    for epoch in range(start_epoch, epochs):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            recons, targets, mu, log_var = net(inputs)
            loss = criterion(recons, targets, mu, log_var)
            writer.add_scalar('Loss_train', loss, epoch)  # tensorboard train loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f'
                         % (train_loss / (batch_idx + 1)))

        scheduler.step()

    print('Saving..')
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, '../muse/checkpoints/{}.pth'.format(args.model_name))


if __name__ == '__main__':
    main()
