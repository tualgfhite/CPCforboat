## Utilities
import argparse
import os
import time
from timeit import default_timer as timer

## Libraries
import numpy as np
## Torch
import torch
import torch.optim as optim
from torch.utils import data

from src.data_reader.dataset import RawDatasetboatClass
## Custrom Imports
from src.logger import setup_logs
from src.model.model import boatClassifier, cpc2
from src.prediction import prediction_boat
from src.test import test_boat
from src.training import train_boat, result

############ Control Center and Hyperparameter ###############
run_name = "cpc" + time.strftime("-%Y-%m-%d_%H_%M_%S")
print(run_name)


class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 128
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.delta = 1

    def state_dict(self):
        self.optimizer.state_dict()

    def step(self):
        """Step by the inner optimizer"""
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr


def main():
    #配置参数
    parser = argparse.ArgumentParser(description='torch cpc')
    parser.add_argument('--datadir', required=True)
    parser.add_argument('--logging-dir', required=True,
                        help='model save directory')
    parser.add_argument('--model-path')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--n-warmup-steps', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--audio_window', type=int, default=16384,
                        help='window length to sample from each utterance')
    parser.add_argument('--frame_window', type=int, default=1)
    parser.add_argument('--boat_num', type=int, default=3)
    parser.add_argument('--timestep', type=int, default=10)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('use_cuda is', use_cuda)
    global_timer = timer()  # global timer
    logger = setup_logs(args.logging_dir, run_name)  # setup logs
    device = torch.device("cuda" if use_cuda else "cpu")
    #加载model loader
    cpc_model = cpc2(args.timestep, args.batch_size, args.audio_window).to(device)
    checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    cpc_model.load_state_dict(checkpoint['state_dict'])
    for param in cpc_model.parameters():
        param.requires_grad = False
    boat_model = boatClassifier(args.boat_num).to(device)
    ## Loading the dataset
    params = {'num_workers': 4,
              'pin_memory': False} if use_cuda else {}
    logger.info('===> loading train, test and eval dataset')
    training_set = RawDatasetboatClass(args.datadir, 'train')
    test_set = RawDatasetboatClass(args.datadir, 'test')
    eval_set = RawDatasetboatClass(args.datadir, 'eval')
    train_loader = data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True,
                                   **params)  # set shuffle to True
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                  **params)  # set shuffle to False
    eval_loader = data.DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, **params)  # set shuffle to False
    # nanxin optimizer  
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, boat_model.parameters()),
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        args.n_warmup_steps)
    model_params = sum(p.numel() for p in boat_model.parameters() if p.requires_grad)
    logger.info('### Model summary below###\n {}\n'.format(str(boat_model)))
    logger.info('===> Model total parameter: {}\n'.format(model_params))
    #train
    best_acc = 0
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train_boat(args.log_interval, cpc_model, boat_model, device, train_loader, optimizer, epoch)
        val_acc, val_loss = test_boat(cpc_model, boat_model, device, test_loader)

        # 保存模型
        if val_acc > best_acc:
            best_acc = max(val_acc, best_acc)
            # if val_loss < best_loss:
            # best_loss = min(val_loss, best_loss)
            result(args.logging_dir, run_name, {
                'epoch': epoch + 1,
                'test_acc': val_acc,
                'state_dict': boat_model.state_dict(),
                'test_loss': val_loss,
                'optimizer': optimizer.state_dict(),
            })
            best_epoch = epoch + 1
        elif epoch - best_epoch > 2:
            optimizer.increase_delta()
            best_epoch = epoch + 1
        logger.info("#### End epoch {}/{}".format(epoch, args.epochs))
    ## prediction 
    logger.info('===> loading best model for prediction')
    checkpoint = torch.load(os.path.join(args.logging_dir, run_name + '-model_best.pth'))
    boat_model.load_state_dict(checkpoint['state_dict'])

    prediction_boat(cpc_model, boat_model, device, eval_loader, args.frame_window)
    ## end
    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))


if __name__ == '__main__':
    main()
