import itertools
import json
import logging
import math
import os
from collections import OrderedDict

import copy

import torch
import random
from torch import nn, optim
from torch.nn.parallel.data_parallel import DataParallel

from tqdm import tqdm
from theconf import Config as C, ConfigArgumentParser

from common import get_logger
from data import get_dataloaders, Get_DataLoaders_Epoch,Get_DataLoaders_Epoch_s, get_val_test_dataloader, change_aug_type
from lr_scheduler import adjust_learning_rate_resnet
from metrics import accuracy, Accumulator
from networks import get_model, num_class
from warmup_scheduler import GradualWarmupScheduler
from augmentations import aug_ohl_list

from common import add_filehandler
from smooth_ce import SmoothCrossEntropyLoss

from itertools import cycle

logger = get_logger('RandAugment')
logger.setLevel(logging.INFO)

def run_epoch(model, loader, loss_fn, optimizer, desc_default='', epoch=0, writer=None, verbose=1, scheduler=None):
    tqdm_disable = bool(os.environ.get('TASK_NAME', ''))    # KakaoBrain Environment
    if verbose:
        loader = tqdm(loader, disable=tqdm_disable)
        loader.set_description('[%s %04d/%04d]' % (desc_default, epoch, C.get()['epoch']))

    metrics = Accumulator()
    cnt = 0
    total_steps = len(loader)
    steps = 0
    for data, label in loader:
        steps += 1
        data, label = data.cuda(), label.cuda()

        if optimizer:
            optimizer.zero_grad()

        preds = model(data)
        loss = loss_fn(preds, label)

        if optimizer:
            loss.backward()
            if C.get()['optimizer'].get('clip', 5) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), C.get()['optimizer'].get('clip', 5))
            optimizer.step()

        top1, top5 = accuracy(preds, label, (1, 5))
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

        if scheduler is not None:
            scheduler.step(epoch - 1 + float(steps) / total_steps)

        del preds, loss, top1, top5, data, label

    if tqdm_disable:
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

def run_epoch_train(model, loaders, val_loader, loss_fn, optimizer,optimizer_aug, aug_param, desc_default='',explore_ratio = 0, w0s_at=[],w0s_mt=[], ops_num = 2, epoch=0, writer=None, verbose=1, scheduler=None,dict_reward={}):
    tqdm_disable = bool(os.environ.get('TASK_NAME', ''))    # KakaoBrain Environment
    if verbose:
        loader_t = tqdm(zip(*loaders), disable=tqdm_disable)
        loader_t.set_description('[%s %04d/%04d]' % (desc_default, epoch, C.get()['epoch']))
    metrics = Accumulator()
    cnt = 0
    total_steps = len(loaders[0])
    print()
    print(loaders[0].dataset.transform.transforms[0])
    print()
    print(loaders[1].dataset.transform.transforms[0])
    print()
    steps = 0
    val_loader = cycle(val_loader)
    w0s_avg = []
    w0s_max = []
    print(torch.nn.Softmax()(aug_param))
    print(w0s_at)
    print(w0s_mt)
    for data_bacth in loader_t:
        # print(torch.nn.Softmax()(aug_param))
        # print(torch.nn.Softmax()(aug_param))
        # model.load_state_dict(model_traget.state_dict())
        data_val, label_val = next(val_loader)
        data_val, label_val = data_val.cuda(), label_val.cuda()
        preds_val = model(data_val)
        loss_V = loss_fn(preds_val, label_val).sum()
        grads_V = torch.autograd.grad(loss_V, (model.parameters()))
        grad_ls = []
        gip_ls =torch.zeros(len(loaders))

        del data_val, label_val, preds_val, loss_V

        if optimizer:
            optimizer.zero_grad()
        if optimizer_aug:
            optimizer_aug.zero_grad()

        for idl in range(len(data_bacth)):
            data, label = data_bacth[idl]
            data_train, label_train = data.cuda(), label.cuda()
            preds_train = model(data_train)
            loss_train = loss_fn(preds_train, label_train)
            loss_T = torch.sum(loss_train)
            if idl == 0:
                preds_train0, label_train0, loss_T0 = preds_train, label_train, loss_T
            grads_T = torch.autograd.grad(loss_T, (model.parameters()))
            #negative of gradient
            gip_ls[idl] = sum([torch.sum(gt*gv) for gt, gv in zip(grads_T, grads_V)]).data
            grad_ls.append(grads_T)
            del data_train, label_train, loss_train, preds_train,loss_T

        del grads_V, grads_T
        w = (gip_ls - torch.mean(gip_ls)) / torch.std(gip_ls)
        # print(w)
        # w_tilde = torch.clamp(gip_ls,min=0)
        # norm_c = torch.sum(w_tilde)
        # if norm_c != 0:
        #     w = w_tilde / norm_c
        # else:
        #     w = w_tilde
        max_id = torch.max(w, 0)[1].data.item()
        w0 = w[0].data.item()
        if w0 >=0:
            w0s_avg.append(1)
        else:
            w0s_avg.append(0)
        if max_id == 0:
            w0s_max.append(1)
        else:
            w0s_max.append(0)
        for idl in range(1,len(loaders)):
            idxs = loaders[idl].dataset.transform.transforms[0].idxs
            # print((idxs,w[idl].data.item() - w0))
            # trace_prob(op_params, op_ids)
            trace_loss = -1 * trace_prob(aug_param, idxs) * (w[idl].data.item()- w0)
            trace_loss.backward()
        optimizer_aug.step()

        grad_T = grad_ls[max_id]
        for gt, p in zip(grad_T, model.parameters()):
            p.grad = gt.data
        if optimizer:
            if C.get()['optimizer'].get('clip', 5) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), C.get()['optimizer'].get('clip', 5))
            optimizer.step()

        top1, top5 = accuracy(preds_train0, label_train0, (1, 5))
        metrics.add_dict({
            'loss': loss_T0.item() * len(data),
            'top1': top1.item() * len(data),
            'top5': top5.item() * len(data),
        })
        cnt += len(data)
        if verbose:
            postfix = metrics / cnt
            if optimizer:
                postfix['lr'] = optimizer.param_groups[0]['lr']
            loader_t.set_postfix(postfix)

        if scheduler is not None:
            # print(epoch - 1 + float(steps) / total_steps)
            # print(optimizer.param_groups[0]['lr'])
            scheduler.step(epoch - 1 + float(steps) / total_steps)
            # print(optimizer.param_groups[0]['lr'])

        del top1, top5
        del gip_ls, grad_ls
        for idl in range(1,len(loaders)):
            if random.random() > explore_ratio:
                tmp_type = (ops_num, select_op(aug_param, ops_num))
            else:
                tmp_type = (ops_num, select_op(torch.zeros(len(aug_ohl_list)), ops_num))
            # print(tmp_type)
            change_aug_type(loaders[idl], tmp_type)
        steps += 1
    w0s_at.append(sum(w0s_avg)/len(w0s_avg))
    w0s_mt.append(sum(w0s_max)/len(w0s_max))
    if tqdm_disable:
        if optimizer:
            logger.info('[%s %03d/%03d] %s lr=%.6f', desc_default, epoch, C.get()['epoch'], metrics / cnt, optimizer.param_groups[0]['lr'])
        else:
            logger.info('[%s %03d/%03d] %s', desc_default, epoch, C.get()['epoch'], metrics / cnt)
    logger.info('[%s %03d/%03d] %s', desc_default, epoch, C.get()['epoch'], metrics / cnt)
    metrics /= cnt
    if optimizer:
        metrics.metrics['lr'] = optimizer.param_groups[0]['lr']
    if optimizer_aug:
        print("param learning rate")
        print(optimizer_aug.param_groups[0]['lr'])
    if verbose:
        for key, value in metrics.items():
            writer.add_scalar(key, value, epoch)
    return metrics
softmax = torch.nn.Softmax()

def select_op(op_params, num_ops):
    prob = softmax(op_params)
    op_ids = torch.multinomial(prob, 2).tolist()
    return op_ids

def trace_prob(op_params, op_ids):
    probs = softmax(op_params)
    tp = 1
    for idx in op_ids:
        tp = tp * probs[idx]
    return tp

def train_and_eval(tag, dataroot, loader_num = 6, test_ratio=0.1, ops_num = 2, explore_ratio = 0, param_lr = 0.05, reporter=None, metric='last', save_path=None, only_eval=False):
    if not reporter:
        reporter = lambda **kwargs: 0

    max_epoch = C.get()['epoch']
    aug_length = len(aug_ohl_list)
    aug_param = torch.nn.Parameter(torch.zeros(aug_length,requires_grad=True),requires_grad=True)
    # if C.get()['optimizer']['type'] == 'sgd':
    #     optimizer_aug = optim.SGD(
    #         (aug_param,),
    #         lr=0.01,
    #         momentum=C.get()['optimizer'].get('momentum', 0.9),
    #         weight_decay=C.get()['optimizer']['decay'],
    #         nesterov=C.get()['optimizer']['nesterov']
    #     )
    # else:
    #     raise ValueError('invalid optimizer type=%s' % C.get()['optimizer']['type'])
    optimizer_aug = torch.optim.Adam((aug_param,),lr=param_lr, betas=(0.5, 0.999))

    trainsampler, validloader, testloader_ = get_val_test_dataloader(C.get()['dataset'], C.get()['batch'], dataroot, test_ratio)

    # create a model & an optimizer
    model = get_model(C.get()['model'], num_class(C.get()['dataset']))
    # model_meta = get_model(C.get()['model'], num_class(C.get()['dataset']))

    lb_smooth = C.get()['optimizer'].get('label_smoothing', 0.0)
    if lb_smooth > 0.0:
        criterion = SmoothCrossEntropyLoss(lb_smooth,reduction='none')
        criterion_val = SmoothCrossEntropyLoss(lb_smooth)
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')
        criterion_val = nn.CrossEntropyLoss()
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

    if C.get()['optimizer'].get('lars', False):
        from torchlars import LARS
        optimizer = LARS(optimizer)
        logger.info('*** LARS Enabled.')

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
        rs['train'] = run_epoch(model, trainloader, criterion_val, None, desc_default='train', epoch=0, writer=writers[0])
        rs['valid'] = run_epoch(model, validloader, criterion_val, None, desc_default='valid', epoch=0, writer=writers[1])
        rs['test'] = run_epoch(model, testloader_, criterion_val, None, desc_default='*test', epoch=0, writer=writers[2])
        for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'valid', 'test']):
            if setname not in rs:
                continue
            result['%s_%s' % (key, setname)] = rs[setname][key]
        result['epoch'] = 0
        return result

    # train loop
    best_top1 = 0
    dict_reward = {}
    w0s_at=[]
    w0s_mt=[]
    for epoch in range(epoch_start, max_epoch + 1):
        
        AugTypes=[(ops_num, select_op(aug_param, ops_num)) for _ in range(loader_num)]
        random.shuffle(trainsampler.indices)
        loaders = Get_DataLoaders_Epoch_s(
            C.get()['dataset'], C.get()['batch'], dataroot, trainsampler, AugTypes, loader_num = len(AugTypes))
        for loader in loaders[1:]:
            print((loader.dataset.transform.transforms[0].n,loader.dataset.transform.transforms[0].idxs))
        model.train()
        rs = dict()
        
        rs['train'] = run_epoch_train(
            model, loaders, validloader, criterion, optimizer,optimizer_aug, aug_param, 
            explore_ratio = explore_ratio, ops_num = ops_num, desc_default='train', epoch=epoch, 
            writer=writers[0], verbose=True, scheduler=scheduler, dict_reward=dict_reward, w0s_at=w0s_at, w0s_mt = w0s_mt)
        model.eval()

        if math.isnan(rs['train']['loss']):
            raise Exception('train loss is NaN.')

        if epoch % 1 == 0 or epoch == max_epoch:
            rs['valid'] = run_epoch(model, validloader, criterion_val, None, desc_default='valid', epoch=epoch, writer=writers[1], verbose=True)
            rs['test'] = run_epoch(model, testloader_, criterion_val, None, desc_default='*test', epoch=epoch, writer=writers[2], verbose=True)

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
                if save_path:
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
                    torch.save({
                        'epoch': epoch,
                        'log': {
                            'train': rs['train'].get_dict(),
                            'valid': rs['valid'].get_dict(),
                            'test': rs['test'].get_dict(),
                        },
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict()
                    }, save_path.replace('.pth', '_e%d_top1_%.3f_%.3f' % (epoch, rs['train']['top1'], rs['test']['top1']) + '.pth'))

    del model

    result['top1_test'] = best_top1
    return result


if __name__ == '__main__':
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--dataroot', type=str, default='/data/private/pretrainedmodels', help='torchvision data folder')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--cv-ratio', type=float, default=0.5)
    parser.add_argument('--explore_ratio', type=float, default=0.2)
    parser.add_argument('--param_lr', type=float, default=0.005)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--loader_num', type=int, default=2)
    parser.add_argument('--ops_num', type=int, default=2)
    parser.add_argument('--only-eval', action='store_true')
    args = parser.parse_args()

    assert (args.only_eval and args.save) or not args.only_eval, 'checkpoint path not provided in evaluation mode.'

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
    result = train_and_eval(args.tag, args.dataroot, loader_num = args.loader_num, param_lr = args.param_lr, ops_num = args.ops_num, test_ratio=args.cv_ratio, save_path=args.save, only_eval=args.only_eval, explore_ratio = args.explore_ratio, metric='test')
    elapsed = time.time() - t

    logger.info('done.')
    logger.info('model: %s' % C.get()['model'])
    logger.info('augmentation: %s' % C.get()['aug'])
    logger.info('\n' + json.dumps(result, indent=4))
    logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logger.info('top1 error in testset: %.4f' % (1. - result['top1_test']))
    logger.info(args.save)
