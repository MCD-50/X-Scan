"""Main file."""
from dataloader import get_train_val_iterators
import configargparse
import progressbar
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from loss import DiceLoss
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
parser.add('--ce_loss_wt', default=5, help='Classification loss wts.', type=float)

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
dice_crit = DiceLoss()
class_crit = nn.CrossEntropyLoss(torch.FloatTensor([1, 5])).cuda()
best_val_loss = 1000000000
for epoch in range(50):
    def run(mode):
        total_loss = AverageValueMeter()
        total_dice_loss = AverageValueMeter()
        total_class_loss = AverageValueMeter()
        if mode == 'train':
            model.train()
        else:
            model.eval()
        bar = progressbar.ProgressBar()
        for batch_idx, data in bar(enumerate(iterators[mode]())):
            input = data['input']
            target_mask = data['mask_target']
            target = data['target']
            # print(target.max(), target.min())
            output = model(Variable(input.cuda(), volatile=mode != 'train'))
            dice_loss, num_indices = dice_crit(
                output['segmentation'], Variable(target_mask.cuda()))
            class_loss = class_crit(
                output['classification'], Variable(target.cuda()))
            loss = class_loss
            if num_indices != 0:
                loss = loss + dice_loss
                total_dice_loss.add(dice_loss[0].data[0])
            total_loss.add(loss[0].data[0])
            total_class_loss.add(class_loss[0].data[0])
            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            gc.collect()
        print("Total loss {}: {}".format(mode, total_loss.value()))
        print("Total dice loss {}: {}".format(mode, total_dice_loss.value()))
        print("Total class loss {}: {}".format(mode, total_class_loss.value()))
        return total_loss.value()[0]
    train_loss = run('train')
    val_loss = run('val')
    if val_loss < best_val_loss:
        checkpoint = {'network': model.module.state_dict(),
                      'epoch': epoch, 'args': options}
        torch.save(checkpoint, 'best_model_{}.pt'.format(options.buckets))
        best_val_loss = val_loss
        print("Best val loss: {} at {}".format(best_val_loss, epoch))
    if (epoch + 1) % 10 == 0:
        for param_gp in optimizer.param_groups:
                param_gp['lr'] *= 0.5
