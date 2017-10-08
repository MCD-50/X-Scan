"""Main file."""
from dataloader import get_train_val_iterators
import configargparse
import progressbar
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from model import Resnet18, Resnet9
from torchnet.meter import AverageValueMeter
import gc


parser = configargparse.get_argument_parser()
parser.add('-c', '--config', required=True, is_config_file=True,
           help='config file path')
parser.add('--n_classes', required=True, type=int, help="Number of classes.")
parser.add('--lr', required=True, type=float, help="Starting learning rate")
parser.add('--nesterov', default=True, help="Nesterov momentum.")
parser.add('--momentum', default=0.9, help="Momentum", type=float)

options, unknown_options = parser.parse_known_args()

iterators = get_train_val_iterators(options)
options.transform = None
# print(iterators())
# dataloader = tqdm(iterable=iterators, ncols=0)

model = Resnet9()
model.cuda()
model = nn.DataParallel(model)
optimizer = optim.SGD(
    model.parameters(),
    lr=options.lr,
    momentum=options.momentum,
    nesterov=options.nesterov)
class_crit = nn.CrossEntropyLoss(torch.FloatTensor([1, 5])).cuda()
best_val_loss = 1000000000
for epoch in range(50):
    def run(mode):
        total_loss = AverageValueMeter()
        if mode == 'train':
            model.train()
        else:
            model.eval()
        bar = progressbar.ProgressBar()
        for batch_idx, data in bar(enumerate(iterators[mode]())):
            input = data['input']
            target = data['target']
            # print(target.max(), target.min())
            output = model(Variable(input.cuda(), volatile=mode != 'train'))
            loss = class_crit(
                output['classification'], Variable(target.cuda()))
            total_loss.add(loss[0].data[0])
            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            gc.collect()
        print("Total loss {}: {}".format(mode, total_loss.value()))
        return total_loss.value()[0]
    train_loss = run('train')
    val_loss = run('val')
    if val_loss < best_val_loss:
        checkpoint = {'network': model.module.state_dict(),
                      'epoch': epoch, 'args': options}
        torch.save(checkpoint, 'best_model_{}.pt'.format(options.buckets))
        best_val_loss = val_loss
        print("Best val loss: {} at {}".format(best_val_loss, epoch))
