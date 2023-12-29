import argparse
import os
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from model import REDNet10, REDNet20, REDNet30
from dataset import Dataset
from utils import AverageMeter
import PIL.Image as pil_image
cudnn.benchmark = True
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='REDNet10', help='REDNet10, REDNet20, REDNet30')
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--jpeg_quality', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--use_fast_loader', action='store_true')
    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    torch.manual_seed(opt.seed)

    if opt.arch == 'REDNet10':
        model = REDNet10()
    elif opt.arch == 'REDNet20':
        model = REDNet20()
    elif opt.arch == 'REDNet30':
        model = REDNet30()
    else:
        raise Exception('No such architecture')

    model = model.to(device)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    dataset = Dataset(opt.images_dir, opt.patch_size, opt.jpeg_quality, opt.use_fast_loader)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.threads,
                            pin_memory=True,
                            drop_last=False)

    for epoch in range(opt.num_epochs):
        epoch_losses = AverageMeter()

        with tqdm(total=(len(dataset) - len(dataset) % opt.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, opt.num_epochs))
            for data in dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                gap = criterion(inputs, labels)

                preds = model(inputs)
                loss = criterion(preds, labels-inputs)
                # loss_2 = criterion(preds, inputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.update(loss.item(), len(inputs))

                if epoch % 10 == 1:
                    preds, labels, inputs = preds[0], labels[0], inputs[0]

                    preds = preds.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
                    output = pil_image.fromarray(preds, mode='RGB')
                    output.save(os.path.join(opt.outputs_dir, f"{epoch}_gap.png"))

                    preds = inputs + preds
                    preds = preds.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
                    output = pil_image.fromarray(preds, mode='RGB')
                    output.save(os.path.join(opt.outputs_dir, f"{epoch}_preds.png"))

                    labels = labels.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
                    output = pil_image.fromarray(labels, mode='RGB')
                    output.save(os.path.join(opt.outputs_dir, f"{epoch}_labels.png"))

                    inputs = inputs.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
                    output = pil_image.fromarray(inputs, mode='RGB')
                    output.save(
                        os.path.join(opt.outputs_dir, f"{epoch}_inputs.png"))

                _tqdm.set_postfix(loss='{:.6f}|{:.6f}'.format(epoch_losses.val, gap.item()))
                _tqdm.update(len(inputs))

    torch.save(model.state_dict(), os.path.join(opt.outputs_dir, '{}_epoch_last.pth'.format(opt.arch)))
