"""
Training scripts.

Authors: Hongjie Fang.
"""
import os
import yaml
import torch
import logging
import argparse
import warnings
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils.logger import ColoredLogger
from utils.builder import ConfigBuilder
from utils.constants import LOSS_INF
from utils.functions import display_results, to_device
from time import perf_counter
torch.multiprocessing.set_sharing_strategy('file_system')

logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)
warnings.simplefilter("ignore", UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type = int, default = 0, help = 'cuda id')
parser.add_argument(
    '--cfg', '-c', 
    default = os.path.join('configs', 'default.yaml'), 
    help = 'path to the configuration file', 
    type = str
)
args = parser.parse_args()
cfg_filename = args.cfg

cuda_id = "cuda:" + str(args.cuda)

with open(cfg_filename, 'r') as cfg_file:
    cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)

builder = ConfigBuilder(**cfg_params)

tensorboard_log = builder.get_tensorboard_log()

logger.info('Building models ...')

model = builder.get_model()

if builder.multigpu():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
else:
    device = torch.device(cuda_id if torch.cuda.is_available() else "cpu")
    if device == torch.device('cpu'):
        raise EnvironmentError('No GPUs, cannot initialize multigpu training.')
    model.to(device)

logger.info('Building dataloaders ...')
train_dataloader = builder.get_dataloader(split = 'train')
test_dataloader = builder.get_dataloader(split = 'test')
test_real_dataloader = builder.get_dataloader(dataset_params={"test": {"type": "cleargrasp-syn", "data_dir": "cleargrasp", "image_size": (320, 240),\
     "use_augmentation": False, "depth_min": 0.0, "depth_max": 10.0,  "depth_norm": 1.0}}, split = 'test')

logger.info('Checking checkpoints ...')
start_epoch = 0
max_epoch = builder.get_max_epoch()
stats_dir = builder.get_stats_dir()
checkpoint_file = os.path.join(stats_dir, 'checkpoint.tar')
if os.path.isfile(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    checkpoint_metrics = checkpoint['metrics']
    checkpoint_loss = checkpoint['loss']
    logger.info("Checkpoint {} (epoch {}) loaded.".format(checkpoint_file, start_epoch))

logger.info('Building optimizer and learning rate schedulers ...')
resume = (start_epoch > 0)
optimizer = builder.get_optimizer(model, resume = resume, resume_lr = 1e-4)
lr_scheduler = builder.get_lr_scheduler(optimizer, resume = resume, resume_epoch = (start_epoch - 1 if resume else None))

if builder.multigpu():
    model = nn.DataParallel(model)

criterion = builder.get_criterion()
metrics = builder.get_metrics()


def train_one_epoch(epoch):
    logger.info('Start training process in epoch {}.'.format(epoch + 1))
    model.train()
    losses = []
    with tqdm(train_dataloader) as pbar:
        for data_dict in pbar:
            optimizer.zero_grad()
            data_dict = to_device(data_dict, device)

            res = model(data_dict['rgb'], data_dict['depth'])
            n, h, w = data_dict['depth'].shape
            data_dict['pred'] = res.view(n, h, w)


            loss_dict = criterion(data_dict)
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            if 'smooth' in loss_dict.keys():
                pbar.set_description('Epoch {}, loss: {:.8f}, smooth loss: {:.8f}'.format(epoch + 1, loss.item(), loss_dict['smooth'].item()))
            else:
                pbar.set_description('Epoch {}, loss: {:.8f}'.format(epoch + 1, loss.item()))
            losses.append(loss.mean().item())
    mean_loss = np.stack(losses).mean()
    logger.info('Finish training process in epoch {}, mean training loss: {:.8f}'.format(epoch + 1, mean_loss))
    return mean_loss


def test_one_epoch(dataloader, epoch):
    logger.info('Start testing process in epoch {}.'.format(epoch + 1))
    model.eval()
    metrics.clear()
    running_time = []
    losses = []
    with tqdm(dataloader) as pbar:
        for data_dict in pbar:
            data_dict = to_device(data_dict, device)
            with torch.no_grad():
                time_start = perf_counter()
                res = model(data_dict['rgb'], data_dict['depth'])
                n, h, w = data_dict['depth'].shape
                data_dict['pred'] = res.view(n, h, w)
                time_end = perf_counter()
                loss_dict = criterion(data_dict)
                loss = loss_dict['loss']
                _ = metrics.evaluate_batch(data_dict, record = True)
            duration = time_end - time_start
            if 'smooth' in loss_dict.keys():
                pbar.set_description('Epoch {}, loss: {:.8f}, smooth loss: {:.8f}'.format(epoch + 1, loss.item(), loss_dict['smooth'].item()))
            else:
                pbar.set_description('Epoch {}, loss: {:.8f}'.format(epoch + 1, loss.item()))
            losses.append(loss.item())
            running_time.append(duration)
    mean_loss = np.stack(losses).mean()
    avg_running_time = np.stack(running_time).mean()
    logger.info('Finish testing process in epoch {}, mean testing loss: {:.8f}, average running time: {:.4f}s'.format(epoch + 1, mean_loss, avg_running_time))
    metrics_result = metrics.get_results()
    metrics.display_results()
    return mean_loss, metrics_result


def train(start_epoch):
    if start_epoch != 0:
        min_loss = checkpoint_loss
        min_loss_epoch = start_epoch
        display_results(checkpoint_metrics, logger)
    else:
        min_loss = LOSS_INF
        min_loss_epoch = None
    for epoch in range(start_epoch, max_epoch):
        logger.info('--> Epoch {}/{}'.format(epoch + 1, max_epoch))
        train_loss = train_one_epoch(epoch)
        real_test_loss, metrics_result = test_one_epoch(test_dataloader, epoch)

        syn_test_loss, _ = test_one_epoch(test_real_dataloader, epoch)

        
        tensorboard_log.add_scalar("train_mean_loss", train_loss, epoch)
        tensorboard_log.add_scalar("real_mean_loss", real_test_loss, epoch)
        tensorboard_log.add_scalar("syn_mean_loss", syn_test_loss, epoch)

        criterion.step()
        save_dict = {
            'epoch': epoch + 1,
            'model_state_dict': model.module.state_dict() if builder.multigpu() else model.state_dict(),
            'loss': real_test_loss,
            'metrics': metrics_result
        }
        torch.save(save_dict, os.path.join(stats_dir, 'checkpoint-epoch{}.tar'.format(epoch)))
        if real_test_loss < min_loss:
            min_loss = real_test_loss
            min_loss_epoch = epoch + 1
            torch.save(save_dict, os.path.join(stats_dir, 'checkpoint.tar'.format(epoch)))
    logger.info('Training Finished. Min testing loss: {:.6f}, in epoch {}'.format(min_loss, min_loss_epoch))
    tensorboard_log.close()


if __name__ == '__main__':
    train(start_epoch = start_epoch)
