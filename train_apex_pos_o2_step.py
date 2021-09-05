import itertools
import json
import logging
import math
import os
from collections import OrderedDict
import numpy as np

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel.data_parallel import DataParallel

from tqdm import tqdm
from theconf import Config as C, ConfigArgumentParser

from common import get_logger
from data import get_dataloaders
from lr_scheduler import adjust_learning_rate_resnet
from metrics import accuracy, Accumulator
from networks import get_model_np, num_class
from warmup_scheduler import GradualWarmupScheduler

from common import add_filehandler
from smooth_ce import SmoothCrossEntropyLoss

import apex
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier

logger = get_logger('RandAugment')
logger.setLevel(logging.INFO)

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.worldsize
    return rt
def run_epoch(model, loader, loss_fn, optimizer,args = None,  desc_default='', epoch=0, writer=None, verbose=1, scheduler=None):
    tqdm_disable = bool(os.environ.get('TASK_NAME', ''))    # KakaoBrain Environment
    if verbose:
        loader = tqdm(loader, disable=tqdm_disable)
        loader.set_description('[%s %04d/%04d]' % (desc_default, epoch, C.get()['epoch']))

    metrics = Accumulator()
    cnt = 0
    total_steps = len(loader)
    steps = 0
    for data, label in loader:
        #if steps == 0:
        #    print(label)
        steps += 1
        data, label = data.cuda(), label.cuda()

        if optimizer:
            optimizer.zero_grad()

        preds = model(data)
        loss = loss_fn(preds, label)

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
        else:
            reduced_loss = loss.data


        if optimizer:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            # loss.backward()
            if C.get()['optimizer'].get('clip', 5) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), C.get()['optimizer'].get('clip', 5))
            optimizer.step()

        top1, top5 = accuracy(preds, label, (1, 5))
        if args.distributed:
            top1 = reduce_tensor(top1)
            top5 = reduce_tensor(top5)
        metrics.add_dict({
            'loss': loss.item() * len(data),
            'top1': top1.item() * len(data),
            'top5': top5.item() * len(data),
        })
        cnt += len(data)
        if verbose:
            postfix = metrics / cnt
            if optimizer:
                postfix['lr'] = optimizer.param_groups[0]['lr']
            loader.set_postfix(postfix)

        #print(optimizer.param_groups[0]['lr'])
        #if scheduler is not None:
        #    scheduler.step(epoch - 1 + float(steps) / total_steps)

        del preds, loss, top1, top5, data, label
    if scheduler is not None:
        scheduler.step()
    if tqdm_disable and args.local_rank == 0:
        if optimizer:
            logger.info('[%s %03d/%03d] %s lr=%.6f', desc_default, epoch, C.get()['epoch'], metrics / cnt, optimizer.param_groups[0]['lr'])
        else:
            logger.info('[%s %03d/%03d] %s', desc_default, epoch, C.get()['epoch'], metrics / cnt)
    logger.info('[%s %03d/%03d] %s', desc_default, epoch, C.get()['epoch'], metrics / cnt)
    metrics /= cnt
    if optimizer:
        metrics.metrics['lr'] = optimizer.param_groups[0]['lr']
    if verbose:
        for key, value in metrics.items():
            writer.add_scalar(key, value, epoch)
    return metrics


def train_and_eval(tag, dataroot, test_ratio=0.0, cv_fold=0, reporter=None, metric='last', save_path=None, only_eval=False, reduct_factor=1.0, args = None):
    if not reporter:
        reporter = lambda **kwargs: 0

    max_epoch = C.get()['epoch']
    trainsampler, trainloader, validloader, testloader_ = get_dataloaders(C.get()['dataset'], C.get()['batch'], dataroot, test_ratio, split_idx=cv_fold)

    if args.distributed:
        train_set = trainloader.dataset
        test_set = testloader_.dataset
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=C.get()['batch']//args.worldsize, sampler=train_sampler, drop_last=True, num_workers=32)
        testloader_ = torch.utils.data.DataLoader(test_set, batch_size=C.get()['batch']//args.worldsize, shuffle=False,sampler=test_sampler, num_workers=32, pin_memory=True, drop_last=False)
    # create a model & an optimizer
    model = get_model_np(C.get()['model'], num_class(C.get()['dataset']))


    lb_smooth = C.get()['optimizer'].get('label_smoothing', 0.0)
    if lb_smooth > 0.0:
        criterion = SmoothCrossEntropyLoss(lb_smooth)
    else:
        criterion = nn.CrossEntropyLoss()
    if C.get()['optimizer']['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=C.get()['lr'],
            momentum=C.get()['optimizer'].get('momentum', 0.9),
            weight_decay=C.get()['optimizer']['decay'],
            nesterov=C.get()['optimizer']['nesterov']
        )
    else:
        raise ValueError('invalid optimizer type=%s' % C.get()['optimizer']['type'])

    model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
    model = DDP(model, delay_allreduce=True)

    lr_scheduler_type = C.get()['lr_schedule'].get('type', 'cosine')
    if lr_scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=C.get()['epoch'], eta_min=0.)
    elif lr_scheduler_type == 'resnet':
        scheduler = adjust_learning_rate_resnet(optimizer)
    else:
        raise ValueError('invalid lr_schduler=%s' % lr_scheduler_type)

    if C.get()['lr_schedule'].get('warmup', None):
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=C.get()['lr_schedule']['warmup']['multiplier'],
            total_epoch=C.get()['lr_schedule']['warmup']['epoch'],
            after_scheduler=scheduler
        )
    
    if not tag:
        from RandAugment.metrics import SummaryWriterDummy as SummaryWriter
        logger.warning('tag not provided, no tensorboard log.')
    else:
        from tensorboardX import SummaryWriter
    writers = [SummaryWriter(log_dir='./logs/%s/%s' % (tag, x)) for x in ['train', 'valid', 'test']]

    result = OrderedDict()
    epoch_start = 1
    if save_path and os.path.exists(save_path):
        logger.info('%s file found. loading...' % save_path)
        data = torch.load(save_path)
        if 'model' in data or 'state_dict' in data:
            key = 'model' if 'model' in data else 'state_dict'
            logger.info('checkpoint epoch@%d' % data['epoch'])
            if not isinstance(model, DataParallel):
                model.load_state_dict({k.replace('module.', ''): v for k, v in data[key].items()})
            else:
                model.load_state_dict({k if 'module.' in k else 'module.'+k: v for k, v in data[key].items()})
            optimizer.load_state_dict(data['optimizer'])
            if data['epoch'] < C.get()['epoch']:
                epoch_start = data['epoch']
            else:
                only_eval = True
        else:
            model.load_state_dict({k: v for k, v in data.items()})
        del data
    else:
        logger.info('"%s" file not found. skip to pretrain weights...' % save_path)
        if only_eval:
            logger.warning('model checkpoint not found. only-evaluation mode is off.')
        only_eval = False

    if only_eval:
        logger.info('evaluation only+')
        model.eval()
        rs = dict()
        rs['train'] = run_epoch(model, trainloader, criterion, None, args = args, desc_default='train', epoch=0, writer=writers[0])
        rs['valid'] = run_epoch(model, validloader, criterion, None, args = args, desc_default='valid', epoch=0, writer=writers[1])
        rs['test'] = run_epoch(model, testloader_, criterion, None, args = args, desc_default='*test', epoch=0, writer=writers[2])
        for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'valid', 'test']):
            if setname not in rs:
                continue
            result['%s_%s' % (key, setname)] = rs[setname][key]
        result['epoch'] = 0
        return result

    # train loop
    best_top1 = 0
    flag_load = 1
    if C.get()['aug'] == 'rwaug_t0':
        aug_pos = 0
    else:
        aug_pos = 2
    print(aug_pos)
    for epoch in range(epoch_start, max_epoch + 1):
        if args.load_tp == 'none':
            break
        else:
            if flag_load == 1:
                prob_dict = np.load(args.load_tp,allow_pickle=True).item()
                dis_ps = prob_dict['dis_ps']
                max_probs = prob_dict['w0s_mt']
                print((len(dis_ps), len(max_probs)))
                th_ls = max_probs
                flag_load = 0
            th_epoch = th_ls[int(epoch/((max_epoch + 0.1)/len(th_ls)))]
            trainloader.dataset.transform.transforms[aug_pos].p = dis_ps[int(epoch/((max_epoch + 0.1)/len(dis_ps)))]
            print(trainloader.dataset.transform.transforms[aug_pos].p)
        trainloader.dataset.transform.transforms[aug_pos].th = th_epoch
        print(trainloader.dataset.transform.transforms[aug_pos].th)
        print(trainloader.dataset.transform.transforms)
        model.train()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        rs = dict()
        rs['train'] = run_epoch(model, trainloader, criterion, optimizer, args = args, desc_default='train', epoch=epoch, writer=writers[0], verbose=True, scheduler=scheduler)
        model.eval()

        if math.isnan(rs['train']['loss']):
            raise Exception('train loss is NaN.')

        if epoch % 1 == 0 or epoch == max_epoch:
            rs['valid'] = run_epoch(model, validloader, criterion, None, args = args, desc_default='valid', epoch=epoch, writer=writers[1], verbose=True)
            rs['test'] = run_epoch(model, testloader_, criterion, None, args = args, desc_default='*test', epoch=epoch, writer=writers[2], verbose=True)

            if metric == 'last' or rs[metric]['top1'] > best_top1:
                if metric != 'last':
                    best_top1 = rs[metric]['top1']
                for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'valid', 'test']):
                    result['%s_%s' % (key, setname)] = rs[setname][key]
                result['epoch'] = epoch

                writers[1].add_scalar('valid_top1/best', rs['valid']['top1'], epoch)
                writers[2].add_scalar('test_top1/best', rs['test']['top1'], epoch)

                reporter(
                    loss_valid=rs['valid']['loss'], top1_valid=rs['valid']['top1'],
                    loss_test=rs['test']['loss'], top1_test=rs['test']['top1']
                )

                # save checkpoint
                if save_path and args.local_rank == 0:
                    logger.info('save model@%d to %s' % (epoch, save_path))
                    torch.save({
                        'epoch': epoch,
                        'log': {
                            'train': rs['train'].get_dict(),
                            'valid': rs['valid'].get_dict(),
                            'test': rs['test'].get_dict(),
                        },
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict()
                    }, save_path)
                    
    del model

    result['top1_test'] = best_top1
    return result


if __name__ == '__main__':
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--dataroot', type=str, default='/root/imagenet', help='torchvision data folder')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--rf', type=float, default=2.0)
    parser.add_argument('--cv-ratio', type=float, default=0.0)
    parser.add_argument('--mul', type=float, default=1)
    parser.add_argument('--sqrt', type=float, default=1)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--load_tp', type=str, default='none')
    parser.add_argument('--only-eval', action='store_true')
    parser.add_argument("--local_rank", default=0, type=int)

    args = parser.parse_args()

    assert (args.only_eval and args.save) or not args.only_eval, 'checkpoint path not provided in evaluation mode.'

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.worldsize = int(os.environ['WORLD_SIZE'])
    print(args.distributed)
    if args.distributed:
        # FOR DISTRIBUTED:  Set the device according to local_rank.
        torch.cuda.set_device(args.local_rank)

        # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
        # environment variables, and requires that you use init_method=`env://`.
        torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')

    torch.backends.cudnn.benchmark = True

    if not args.only_eval:
        if args.save:
            logger.info('checkpoint will be saved at %s' % args.save)
        else:
            logger.warning('Provide --save argument to save the checkpoint. Without it, training result will not be saved!')

    if args.save:
        add_filehandler(logger, args.save.replace('.pth', '') + '.log')

    logger.info(json.dumps(C.get().conf, indent=4))

    import time
    t = time.time()
    result = train_and_eval(args.tag, args.dataroot, test_ratio=args.cv_ratio, cv_fold=args.cv, save_path=args.save, only_eval=args.only_eval, metric='test',reduct_factor = args.rf, args = args)
    elapsed = time.time() - t

    logger.info('done.')
    logger.info('model: %s' % C.get()['model'])
    logger.info('augmentation: %s' % C.get()['aug'])
    logger.info('\n' + json.dumps(result, indent=4))
    logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logger.info('top1 error in testset: %.4f' % (1. - result['top1_test']))
    logger.info(args.save)