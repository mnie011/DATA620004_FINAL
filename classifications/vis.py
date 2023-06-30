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
from resnet import ResNet18
from utils import progress_bar, cutmix_data, cutout_data, mixup_data
from loss_utils import CutmixCriterion, CECriterion
from torch.autograd import Variable


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='../cfgs/cls/cutmix.yaml', help='specify the config for training')
    parser.add_argument('--version', type=str, default=None, required=False)
    parser.add_argument('--train_batch_size', type=int, default=None, required=False, help='batch_size for training')
    parser.add_argument('--test_batch_size', type=int, default=None, required=False)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=None, required=False)
    parser.add_argument('--momentum', type=float, default=None, required=False)
    parser.add_argument('--weight_decay', type=float, default=None, required=False)

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def preprocess(version, inputs, targets):
    assert version in ['baseline', 'cut_mix', 'cut_off', 'mix_up']
    target_dict = dict()
    if version == 'cut_mix':
        inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, use_cuda=False)
        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
        target_dict['targets_a'] = targets_a
        target_dict['targets_b'] = targets_b
        target_dict['lam'] = lam
    elif version == 'cut_off':
        inputs, targets = cutout_data(inputs, targets, use_cuda=False)
        target_dict['targets'] = targets
    elif version == 'mix_up':
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, use_cuda=False)
        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
        target_dict['targets_a'] = targets_a
        target_dict['targets_b'] = targets_b
        target_dict['lam'] = lam
    else:
        target_dict['targets'] = targets
    return dict(inputs=inputs, targets=target_dict)


def build_criterion(version):
    assert version in ['baseline', 'cut_mix', 'cut_off', 'mix_up']
    if version == 'cut_mix':
        criterion = CutmixCriterion()
    elif version == 'cut_off':
        criterion = CECriterion()
    elif version == 'mix_up':
        criterion = CutmixCriterion()
    else:
        criterion = CECriterion()
    return criterion


def vis(input, version, index):
    image = transforms.ToPILImage()(input)
    image.save('{}_{}.jpg'.format(index, version))


def main():
    args, cfg = parse_config()

    start_epoch = args.start_epoch
    epochs = args.epochs

    if args.version is None:
        args.version = cfg.VERSION

    if args.train_batch_size is None:
        args.train_batch_size = cfg.TRAIN_BATCH_SIZE
        args.test_batch_size = cfg.TEST_BATCH_SIZE

    if args.lr is None:
        args.lr = cfg.LR
        args.momentum = cfg.MOMENTUM
        args.weight_decay = cfg.WEIGHT_DECAY

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    val_percent = 0.2
    trainset = torchvision.datasets.CIFAR10(
        root='../muse', train=True, download=True, transform=transform_train)
    classes = trainset.classes
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='../muse', train=False, download=True, transform=None)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)



    print('==> Building model..')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    net = ResNet18()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('../muse/checkpoints'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('../muse/checkpoints/r18_{}.pth'.format(args.version))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = build_criterion(args.version)
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    writer = SummaryWriter('./path/{}/log'.format(args.version))

    for epoch in range(start_epoch, epochs):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # preprocessed = preprocess(args.version, inputs, targets)
            # inputs = preprocessed['inputs']
            # target_dict = preprocessed['targets']

            preprocessed = preprocess('cut_mix', inputs, targets)
            vis(preprocessed['inputs'][0], 'cut_mix', 0)
            vis(preprocessed['inputs'][1], 'cut_mix', 1)
            vis(preprocessed['inputs'][2], 'cut_mix', 2)

            preprocessed = preprocess('cut_off', inputs, targets)
            vis(preprocessed['inputs'][0], 'cut_off', 0)
            vis(preprocessed['inputs'][1], 'cut_off', 1)
            vis(preprocessed['inputs'][2], 'cut_off', 2)

            preprocessed = preprocess('mix_up', inputs, targets)
            vis(preprocessed['inputs'][0], 'mix_up', 0)
            vis(preprocessed['inputs'][1], 'mix_up', 1)
            vis(preprocessed['inputs'][2], 'mix_up', 2)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, target_dict)
            writer.add_scalar('Loss_train_{}'.format(args.version), loss, epoch)  # tensorboard train loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | mAP: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        # test
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        test_crit = nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = test_crit(outputs, targets)
                writer.add_scalar('Loss_test_baseline', loss, epoch)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | mAP: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total
        writer.add_scalar('mAP_test_{}'.format(args.version), acc, epoch)  # tensorboard acc
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, '../muse/checkpoints/r18_{}.pth'.format(args.version))
            best_acc = acc

        scheduler.step()


if __name__ == '__main__':
    main()
