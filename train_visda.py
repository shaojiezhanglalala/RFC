import os
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
# from torchnet.meter import AUCMeter

from utils.util import AverageMeter, accuracy, TrackMeter
from utils.util import set_seed

from utils.config import Config, ConfigDict, DictAction
from losses import build_loss
from builder import build_optimizer
from models.build import build_model
from utils.util import format_time, interleave, de_interleave
from builder import build_logger
from datasets import build_divm_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume', type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('--load', type=str, help='Load init weights for fine-tune (default: None)')
    parser.add_argument('--cfgname', default='train', help='specify log_file; for debug use')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu is used')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override the config; e.g., --cfg-options port=10001 k1=a,b k2="[a,b]"'
                             'Note that the quotation marks are necessary and that no white space is allowed.')
    args = parser.parse_args()
    return args


def get_cfg(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # args.config： configs/office_home/src_A/train_src_A.py
        # os.rename(oldname, newname, max_times)
        # dirname=checkpoints/office_home/src_A
        dirname = os.path.dirname(args.config).replace('configs', 'checkpoints', 1)
        # basename=train_src_A.py
        # filename=train_src
        filename = os.path.splitext(os.path.basename(args.config))[0]
        # work_dir = checkpoints/office_home/src_A/train_src
        cfg.work_dir = os.path.join(dirname, filename, args.cfgname)
    os.makedirs(cfg.work_dir, exist_ok=True)

    # cfgname
    if args.cfgname is not None:
        cfg.cfgname = args.cfgname
    else:
        cfg.cfgname = os.path.splitext(os.path.basename(args.config))[0]
    assert cfg.cfgname is not None

    # seed
    if args.seed != 0:
        cfg.seed = args.seed
    elif not hasattr(cfg, 'seed'):
        cfg.seed = 42
    set_seed(cfg.seed)

    # resume or load init weights
    if args.resume:
        cfg.resume = args.resume
    if args.load:
        cfg.load = args.load
    assert not (cfg.resume and cfg.load)

    # gpu
    cfg.gpu = args.gpu

    return cfg


def adjust_lr(optimizer, step, tot_steps, gamma=10, power=0.75):
    decay = (1 + gamma * step / tot_steps) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['init_lr'] * decay


def set_optimizer(model, cfg):
    base_params = [v for k, v in model.named_parameters() if 'fc' not in k]
    head_params = [v for k, v in model.named_parameters() if 'fc' in k]
    param_groups = [{'params': base_params, 'lr': cfg.lr * 0.1},
                    {'params': head_params, 'lr': cfg.lr}]
    optimizer = build_optimizer(cfg.optimizer, param_groups)
    for param_group in optimizer.param_groups:
        param_group['init_lr'] = param_group['lr']
    return optimizer


def set_model(cfg):
    model = build_model(cfg.tgt_model)
    model.fc = build_model(cfg.tgt_head)
    model = model.cuda()
    return model


def update_batch_stats(model, flag):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.update_batch_stats = flag


def test(test_loader, model, criterion, epoch, logger, writer, model2=None):
    """ test target """
    model.eval()
    if model2 is not None:
        model2.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top3 = AverageMeter()
    top4 = AverageMeter()
    top5 = AverageMeter()
    all_pred = []

    time1 = time.time()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            logits = model(images)
            if model2 is not None:
                logits2 = model2(images)
                logits = (logits + logits2) / 2
            loss = criterion(logits, labels)

            pred = F.softmax(logits, dim=1)
            all_pred.append(pred.detach())

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc2, acc3, acc4,acc5 = accuracy(logits, labels, topk=(1, 2, 3, 4, 5))
            top1.update(acc1[0], bsz)
            top2.update(acc2[0], bsz)
            top3.update(acc3[0], bsz)
            top4.update(acc4[0], bsz)
            top5.update(acc5[0], bsz)

    all_pred = torch.cat(all_pred)
    mean_ent = (-all_pred * torch.log(all_pred + 1e-5)).sum(dim=1).mean().item() / np.log(all_pred.size(0))

    # writer
    writer.add_scalar(f'Loss/divm_test', losses.avg, epoch)
    writer.add_scalar(f'Entropy/divm_test', mean_ent, epoch)
    writer.add_scalar(f'Acc/divm_test', top1.avg, epoch)
    writer.add_scalar(f'Top5Acc/divm_test', top5.avg, epoch)

    # logger
    time2 = time.time()
    test_time = format_time(time2 - time1)
    logger.info(f'Test at epoch [{epoch}] - test_time: {test_time}, '
                f'test_loss: {losses.avg:.3f}, '
                f'test_entropy: {mean_ent:.3f}, '
                f'test_Acc@1: {top1.avg:.2f},'
                f'test_Acc@2: {top2.avg:.2f},'
                f'test_Acc@3: {top3.avg:.2f},'
                f'test_Acc@4: {top4.avg:.2f},'
                f'test_Acc@5:{top5.avg:.2f}')
    return top1.avg, mean_ent


def test_class_acc(test_loader, model, criterion, it, logger, writer, cfg, model2=None):
    """ test target """
    model.eval()
    if model2 is not None:
        model2.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    all_pred, all_labels = [], []

    time1 = time.time()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            all_labels.append(labels)
            bsz = labels.shape[0]

            # forward
            logits = model(images)
            if model2 is not None:
                logits2 = model2(images)
                logits = (logits + logits2) / 2
            loss = criterion(logits, labels)

            pred = F.softmax(logits, dim=1)
            all_pred.append(pred.detach())

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

    all_labels = torch.cat(all_labels)
    all_pred = torch.cat(all_pred)
    mean_ent = (-all_pred * torch.log(all_pred + 1e-5)).sum(dim=1).mean().item() / np.log(all_pred.size(0))
    pred_max = all_pred.max(dim=1).indices

    # class-wise acc
    class_accs = []
    class_num = []
    class_recall = []
    pred_num = []
    all_eq = pred_max == all_labels
    for c in range(cfg.num_classes):
        mask_c = all_labels == c
        class_num.append(mask_c.sum().item())
        acc_c = all_eq[mask_c].float().mean().item()
        class_accs.append(round(acc_c * 100, 2))
        mask_c_recall = pred_max == c
        pred_num.append(mask_c_recall.sum().item())
        recall = all_eq[mask_c_recall].float().mean().item()
        class_recall.append(round(recall * 100, 2))
    avg_acc = round(sum(class_accs) / len(class_accs), 2)

    # writer
    writer.add_scalar(f'Loss/ft_tgt_test', losses.avg, it)
    writer.add_scalar(f'Entropy/ft_tgt_test', mean_ent, it)
    writer.add_scalar(f'Acc/ft_tgt_test', top1.avg, it)

    # logger
    time2 = time.time()
    test_time = format_time(time2 - time1)
    logger.info(f'Test at iter [{it}] - test_time: {test_time}, '
                f'test_loss: {losses.avg:.3f}, '
                f'test_entropy: {mean_ent:.3f}, '
                f'test_Acc@1: {top1.avg:.2f}')
    logger.info(f'per class acc: {str(class_accs)}, avg_acc: {avg_acc}')
    logger.info(f'per class num: {str(class_num)}')
    logger.info(f'per class recall: {str(class_recall)}')
    logger.info(f'per pred num: {str(pred_num)}')
    return top1.avg, mean_ent, pred_max


def pred_target(test_loader, model, epoch, logger, cfg, model2=None):
    """ get predictions for target samples """
    model.eval()
    if model2 is not None:
        model2.eval()

    all_psl = []
    all_labels = []
    all_pred = []

    time1 = time.time()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            labels = labels.cuda()
            bsz = images.shape[0]

            # forward
            logits = model(images)
            if model2 is not None:
                output2 = model2(images)
                logits = (logits + output2) / 2

            psl = logits.max(dim=1).indices
            pred = F.softmax(logits, dim=1)

            if epoch == 0:
                src_idx = torch.sort(pred, dim=1, descending=True).indices
                for i in range(bsz):
                    pred[i, src_idx[i, cfg.topk:]] = \
                        (1.0 - pred[i, src_idx[i, :cfg.topk]].sum()) / (cfg.num_classes - cfg.topk)

            all_psl.append(psl)
            all_labels.append(labels)
            all_pred.append(pred.detach())
    all_psl = torch.cat(all_psl)
    all_labels = torch.cat(all_labels)
    all_pred = torch.cat(all_pred)
    psl_acc = (all_psl == all_labels).float().mean()

    # logger
    time2 = time.time()
    pred_time = format_time(time2 - time1)
    logger.info(f'Predict target at epoch [{epoch}]: psl_acc: {psl_acc:.2f}, time: {pred_time}')
    return all_psl, all_labels, all_pred


def warmup(warmup_loader, model, optimizer, epoch, logger, cfg):
    batch_time = AverageMeter()
    losses = AverageMeter()

    criterion = nn.CrossEntropyLoss().cuda()
    num_iters = len(warmup_loader)

    model.train()
    t1 = end = time.time()
    for batch_idx, (inputs, labels) in enumerate(warmup_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logger
        if batch_idx % cfg.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f'Epoch [{epoch}][{batch_idx}/{num_iters}] - '
                        f'Batch time: {batch_time.avg:.2f}, '
                        f'lr: {lr:.6f}, '
                        f'loss: {losses.avg:.3f}')

    t2 = time.time()
    epoch_time = format_time(t2 - t1)
    logger.info(f'Epoch [{epoch}] - train_time: {epoch_time}, '
                f'train_loss: {losses.avg:.3f}\n')


def dist_train(warmup_loader_idx, model, optimizer, epoch, logger, cfg, pred_mem):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_kl = AverageMeter()
    losses_ent = AverageMeter()

    num_iters = len(warmup_loader_idx)

    model.train()
    t1 = end = time.time()
    for batch_idx, (images, _, indices) in enumerate(warmup_loader_idx):
        images = images.cuda()
        targets = pred_mem[indices, :]
        bsz = images.shape[0]

        # forward
        logits = model(images)
        pred_tgt = F.softmax(logits, dim=1)
        loss_kl = nn.KLDivLoss(reduction='batchmean')(pred_tgt.log(), targets)

        loss_entropy = (-pred_tgt * torch.log(pred_tgt + 1e-5)).sum(dim=1).mean()
        pred_mean = pred_tgt.mean(dim=0)
        loss_gentropy = torch.sum(-pred_mean * torch.log(pred_mean + 1e-5))
        loss_entropy -= loss_gentropy
        loss = loss_kl + loss_entropy

        # update metric
        losses.update(loss.item(), bsz)
        losses_kl.update(loss_kl.item(), bsz)
        losses_ent.update(loss_entropy.item(), bsz)

        # backward1
        optimizer.zero_grad()
        loss.backward()

        # backward2
        if cfg.lam_mix > 0:
            alpha = 0.3
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(bsz).cuda()
            mixed_images = lam * images + (1 - lam) * images[index, :]
            mixed_targets = (lam * pred_tgt + (1 - lam) * pred_tgt[index, :]).detach()

            update_batch_stats(model, False)
            mixed_logits = model(mixed_images)
            update_batch_stats(model, True)
            mixed_pred_tgt = F.softmax(mixed_logits, dim=1)
            loss_mix_kl = cfg.lam_mix * nn.KLDivLoss(reduction='batchmean')(mixed_pred_tgt.log(), mixed_targets)
            loss_mix_kl.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logger
        if batch_idx % cfg.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Epoch [{epoch}][{batch_idx}/{num_iters}] - '
                f'Batch time: {batch_time.avg:.2f}, '
                f'lr: {lr:.6f}, '
                f'loss_kl: {losses_kl.avg:.3f}, '
                f'loss_ent: {losses_ent.avg:.3f}, '
                f'distill_loss: {losses.avg:.3f}'
            )

    t2 = time.time()
    epoch_time = format_time(t2 - t1)
    logger.info(
        f'Epoch [{epoch}] - train_time: {epoch_time}, '
        f'loss_kl: {losses_kl.avg:.3f}, '
        f'loss_ent: {losses_ent.avg:.3f}, '
        f'distill_loss: {losses.avg:.3f}'
    )


def weight_dist_train(warmup_loader_idx, model, optimizer, epoch, logger, cfg, pred_mem, samples_per_cls):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_kl = AverageMeter()
    losses_ent = AverageMeter()

    num_iters = len(warmup_loader_idx)
    cls_weight = (samples_per_cls.max() - samples_per_cls + 1) / samples_per_cls
    print(cls_weight)

    model.train()
    t1 = end = time.time()
    for batch_idx, (images, _, indices) in enumerate(warmup_loader_idx):
        images = images.cuda()
        targets = pred_mem[indices, :]
        bsz = images.shape[0]

        # forward
        logits = model(images)
        pred_tgt = F.softmax(logits, dim=1)
        with torch.no_grad():
            psl = logits.max(dim=1).indices
            weight = cls_weight[psl]
            weight = weight.detach()

        loss_kl = torch.mean(weight * torch.sum(nn.KLDivLoss(reduction='none')(pred_tgt.log(), targets), dim=1))

        # MI
        loss_entropy = (-pred_tgt * torch.log(pred_tgt + 1e-5)).sum(dim=1).mean()
        pred_mean = pred_tgt.mean(dim=0)
        loss_gentropy = torch.sum(-pred_mean * torch.log(pred_mean + 1e-5))
        loss_entropy -= loss_gentropy
        loss = loss_kl + loss_entropy

        # update metric
        losses.update(loss.item(), bsz)
        losses_kl.update(loss_kl.item(), bsz)
        losses_ent.update(loss_entropy.item(), bsz)

        # backward1
        optimizer.zero_grad()
        loss.backward()

        # backward2
        if cfg.lam_mix > 0:
            alpha = 0.3
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(bsz).cuda()
            mixed_images = lam * images + (1 - lam) * images[index, :]
            mixed_targets = (lam * pred_tgt + (1 - lam) * pred_tgt[index, :]).detach()
            mixed_weight = (lam * weight + (1 - lam) * weight[index]).detach()

            update_batch_stats(model, False)
            mixed_logits = model(mixed_images)
            update_batch_stats(model, True)
            mixed_pred_tgt = F.softmax(mixed_logits, dim=1)
            loss_mix_kl = cfg.lam_mix * torch.mean(mixed_weight * torch.sum(
                nn.KLDivLoss(reduction='none')(mixed_pred_tgt.log(), mixed_targets), dim=1))
            loss_mix_kl.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logger
        if batch_idx % cfg.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Epoch [{epoch}][{batch_idx}/{num_iters}] - '
                f'Batch time: {batch_time.avg:.2f}, '
                f'lr: {lr:.6f}, '
                f'loss_kl: {losses_kl.avg:.3f}, '
                f'loss_ent: {losses_ent.avg:.3f}, '
                f'distill_loss: {losses.avg:.3f}'
            )

    t2 = time.time()
    epoch_time = format_time(t2 - t1)
    logger.info(
        f'Epoch [{epoch}] - train_time: {epoch_time}, '
        f'loss_kl: {losses_kl.avg:.3f}, '
        f'loss_ent: {losses_ent.avg:.3f}, '
        f'distill_loss: {losses.avg:.3f}'
    )


def select_train(sel_loader_idx, model, optimizer, epoch, logger, cfg, pred_mem):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_ce = AverageMeter()
    losses_rce = AverageMeter()

    sel_train_iter = iter(sel_loader_idx)
    num_iters = len(sel_loader_idx)

    model.train()
    t1 = end = time.time()

    for batch_idx in range(num_iters):
        try:
            # weakAug, strongAug, pseudo label, indice_s
            (inputs_s1, inputs_s2), targets_s, indice_s = next(sel_train_iter)
        except:
            assert False

        if len(indice_s) <= 1:
            break

        inputs_s1, inputs_s2 = inputs_s1.cuda(), inputs_s2.cuda()
        targets_s = targets_s.cuda()

        # 计算样本交叉熵损失
        logit_s1 = model(inputs_s1)
        logit_s2 = model(inputs_s2)
        psl_one_hot = F.one_hot(targets_s, cfg.num_classes).float().cuda()
        loss_ce = -torch.mean(torch.sum(F.log_softmax(logit_s1, dim=1) * psl_one_hot, dim=1))
        loss_ce -= torch.mean(torch.sum(F.log_softmax(logit_s2, dim=1) * psl_one_hot, dim=1))
        losses_ce.update(loss_ce)

        # 计算样本的逆交叉熵损失
        pred_s1 = F.softmax(logit_s1, dim=1)
        pred_s2 = F.softmax(logit_s2, dim=1)
        pred = (pred_s1 + pred_s2) / 2
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        psl_one_hot = torch.clamp(psl_one_hot, min=1e-4, max=1.0)
        loss_rce = -torch.mean(torch.sum(psl_one_hot.log() * pred, dim=1))
        losses_rce.update(loss_rce)

        loss = loss_ce + loss_rce
        losses.update(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新样本的软标签
        pred_mem[indice_s, :] = cfg.ema * pred_mem[indice_s, :] + (1 - cfg.ema) * pred.detach()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logger
        if batch_idx % cfg.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Epoch [{epoch}][{batch_idx}/{num_iters}] - '
                f'Batch time: {batch_time.avg:.2f}, '
                f'lr: {lr:.6f}, '
                f'loss: {losses.avg:.3f}, '
                f'loss_ce: {losses_ce.avg:.3f}, '
                f'loss_rce: {losses_rce.avg:.3f}.'
            )

    t2 = time.time()
    epoch_time = format_time(t2 - t1)
    logger.info(
        f'Epoch [{epoch}] - train_time: {epoch_time}, '
        f'loss: {losses.avg:.3f}, '
        f'loss_ce: {losses_ce.avg:.3f}, '
        f'loss_rce: {losses_rce.avg:.3f}.'
    )


def contrast_learning(warmup_loader_idx, model, optimizer, epoch, logger, cfg, writer, fea_bank, score_bank):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_cont_1 = AverageMeter()
    losses_cont_2 = AverageMeter()

    num_iters = len(warmup_loader_idx)
    max_iter = (cfg.epochs - cfg.warmup_epochs) * num_iters
    pre_iter = (epoch - 1 - cfg.warmup_epochs) * num_iters

    model.train()
    t1 = end = time.time()
    for batch_idx, (images, _, indices) in enumerate(warmup_loader_idx):
        images = images.cuda()
        bsz = images.shape[0]

        cur_iter = pre_iter + batch_idx
        alpha = (1 + 10 * cur_iter / max_iter) ** (-cfg.beta)

        # forward
        logits, feat = model(images, req_feat=True)
        pred = F.softmax(logits, dim=1)

        with torch.no_grad():
            feat_norm = F.normalize(feat)
            feat_detach = feat_norm.cpu().detach().clone()

            fea_bank[indices] = feat_detach.detach().clone().cpu()
            score_bank[indices] = pred.detach().clone()

            distance = feat_detach @ fea_bank.T
            _, idx_near = torch.topk(distance, dim=-1, largest=True, k=cfg.K + 1)
            idx_near = idx_near[:, 1:]  # batch x K
            score_near = score_bank[idx_near]  # batch x K x C

        # nn
        softmax_out_un = pred.unsqueeze(1).expand(
            -1, cfg.K, -1
        )  # batch x K x C

        loss = torch.mean(
            (F.kl_div(softmax_out_un, score_near, reduction="none").sum(-1)).sum(1)
        ) # Equal to dot product
        losses_cont_1.update(loss.item(), bsz)

        mask = torch.ones((bsz, bsz))
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag
        copy = pred.T  # .detach().clone()#

        dot_neg = pred @ copy  # batch x batch

        dot_neg = (dot_neg * mask.cuda()).sum(-1)  # batch
        neg_pred = torch.mean(dot_neg)
        loss += alpha * neg_pred

        # update metric
        losses.update(loss.item(), bsz)
        losses_cont_2.update(neg_pred.item(), bsz)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logger
        if batch_idx % cfg.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Epoch [{epoch}][{batch_idx}/{num_iters}] - '
                f'Batch time: {batch_time.avg:.2f}, '
                f'lr: {lr:.6f}, '
                f'loss_cont_1: {losses_cont_1.avg:.3f}, '
                f'loss_cont_2: {losses_cont_2.avg:.3f}, '
                f'loss: {losses.avg:.3f}, '
            )

    t2 = time.time()
    epoch_time = format_time(t2 - t1)
    logger.info(
        f'Epoch [{epoch}] - train_time: {epoch_time}, '
        f'loss_cont_1: {losses_cont_1.avg:.3f}, '
        f'loss_cont_2: {losses_cont_2.avg:.3f}, '
        f'loss: {losses.avg:.3f}'
    )

    writer.add_scalar(f'dist/cont_1_loss', losses_cont_1.avg, epoch)
    writer.add_scalar(f'dist/cont_2_loss', losses_cont_2.avg, epoch)
    writer.add_scalar(f'dist/loss', losses.avg, epoch)


def finetune_train(warmup_loader_idx, model, optimizer, epoch, logger, cfg):
    batch_time = AverageMeter()
    losses = AverageMeter()

    num_iters = len(warmup_loader_idx)

    model.train()
    t1 = end = time.time()
    for batch_idx, (images, _, indices) in enumerate(warmup_loader_idx):
        images = images.cuda()
        bsz = images.shape[0]

        # forward
        logits = model(images)
        pred_tgt = F.softmax(logits, dim=1)

        # MI
        loss_entropy = (-pred_tgt * torch.log(pred_tgt + 1e-5)).sum(dim=1).mean()
        pred_mean = pred_tgt.mean(dim=0)
        loss_gentropy = torch.sum(-pred_mean * torch.log(pred_mean + 1e-5))
        loss_entropy -= loss_gentropy

        # update metric
        losses.update(loss_entropy.item(), bsz)

        # backward1
        optimizer.zero_grad()
        loss_entropy.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logger
        if batch_idx % cfg.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Epoch [{epoch}][{batch_idx}/{num_iters}] - '
                f'Batch time: {batch_time.avg:.2f}, '
                f'lr: {lr:.6f}, '
                f'finetune_loss: {losses.avg:.3f}'
            )

    t2 = time.time()
    epoch_time = format_time(t2 - t1)
    logger.info(
        f'Epoch [{epoch}] - train_time: {epoch_time}, '
        f'finetune_loss: {losses.avg:.3f}'
    )


def eval_train(eval_loader, model):
    model.eval()
    losses = []
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(eval_loader):  # shuffle=False
            inputs = inputs.cuda()
            targets = targets.cuda()  # 源域预测的伪标签

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss)

    losses = torch.cat(losses)
    losses = (losses - losses.min()) / (losses.max() - losses.min())    # 归一化, [0,1]
    losses = losses.cpu()

    # fit a two-component GMM to the loss
    input_loss = losses.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]  # 样本属于平均损失小的高斯模型的概率
    return prob, losses


def obtain_sel_samples_indices(test_loader, model, tgt_psl, loss_mem, epoch, cfg, logger):
    """ get predictions for target samples """
    model.eval()
    all_psl = []
    all_labels = []
    all_pred = []
    num_classes = cfg.num_classes

    time1 = time.time()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            labels = labels.cuda()
            bsz = images.shape[0]

            # forward
            logits = model(images)

            psl = logits.max(dim=1).indices
            pred = F.softmax(logits, dim=1)

            src_idx = torch.sort(pred, dim=1, descending=True).indices

            if epoch == 0:
                for i in range(bsz):
                    pred[i, src_idx[i, cfg.topk:]] = \
                        (1.0 - pred[i, src_idx[i, :cfg.topk]].sum()) / (num_classes - cfg.topk)

            all_psl.append(psl)
            all_labels.append(labels)
            all_pred.append(pred.detach())

    all_psl = torch.cat(all_psl)
    all_labels = torch.cat(all_labels)
    all_pred = torch.cat(all_pred)
    all_entropy = (-all_pred * torch.log(all_pred + 1e-5)).sum(dim=1)

    # 更新样本的loss
    psl_one_hot = F.one_hot(tgt_psl, num_classes).float()
    all_loss = -torch.sum(torch.log(all_pred) * psl_one_hot, dim=1)
    if epoch == 0:
        loss_mem = all_loss.cpu().detach()
    else:
        loss_mem = cfg.ema * loss_mem + (1 - cfg.ema) * all_loss.cpu().detach()

    # obtain class entropy rank
    cls_entropy = torch.zeros(num_classes)
    for i in range(num_classes):
        cls_entropy[i] = all_entropy[torch.where(tgt_psl == i)].mean()
    _, entropy_idx = torch.sort(cls_entropy, dim=0, descending=True)
    cls_entropy_rank = torch.zeros(num_classes)
    for i in range(num_classes):
        cls_entropy_rank[entropy_idx[i]] = i

    # count each class pred number
    samples_per_cls_cur = torch.zeros(num_classes)
    for i in range(num_classes):
        samples_per_cls_cur[i] = (all_psl == i).sum()

    # obtain class pred rank
    pred_idx = torch.sort(samples_per_cls_cur, dim=0).indices
    cls_pre_rank = torch.zeros(num_classes)
    for i in range(num_classes):
        cls_pre_rank[pred_idx[i]] = i
    # obtain class select rank
    cls_idx = torch.sort((cls_pre_rank + cls_entropy_rank), dim=0).indices
    # log all rank results above
    logger.info(f'cls_entropy_rank:{str(entropy_idx.tolist())},\n '
                f'cls_pred_rank:{str(pred_idx.tolist())},\n'
                f'cls_rank:{str(cls_idx.tolist())}.')

    # 获得每个类的伪标签个数
    tgt_psl_num_per_cls = torch.zeros(num_classes)
    for i in range(num_classes):
        tgt_psl_num_per_cls[i] = (tgt_psl == i).sum()

    max_loss = float('inf')
    cls_sel_num = int(cfg.sel_tau * num_classes)
    sel_idx = []
    sel_cls_idx = cls_idx[:cls_sel_num]
    median_num = tgt_psl_num_per_cls[sel_cls_idx].median()
    sel_cls_num = []
    sel_acc_per_cls = []
    for i in sel_cls_idx:
        tmp_loss = loss_mem.clone().detach()
        tmp_loss[torch.where(tgt_psl != i)] = max_loss
        sort_loss_idx_cls = torch.sort(tmp_loss, dim=0)[1]
        sel_num = min(int(cfg.sample_tau * torch.sqrt(median_num * tgt_psl_num_per_cls[i])), int(tgt_psl_num_per_cls[i]))
        sel_idx.append(sort_loss_idx_cls[:sel_num])
        sort_idx = sort_loss_idx_cls[:sel_num]
        sel_acc_cls = 100 * (tgt_psl[sort_idx] == all_labels[sort_idx]).float().sum() / sel_num
        sel_acc_per_cls.append('{:.2f}%'.format(sel_acc_cls))
        sel_cls_num.append(sel_num)
    sel_idx = torch.cat(sel_idx).cpu()

    sel_acc = (tgt_psl[sel_idx] == all_labels[sel_idx]).float().mean().item()
    logger.info(f'epoch:{epoch}, select_number:{len(sel_idx)}, '
                f'select_acc:{sel_acc * 100:.2f}%, '
                f'select_class:{sel_cls_idx.tolist()}, '
                f'select_num_per_class:{str(sel_cls_num)}, '
                f'sel_acc_per_cls:{sel_acc_per_cls}.')
    # logger
    time2 = time.time()
    select_time = format_time(time2 - time1)
    logger.info(f'obtain_select_samples_idx_time:{select_time}.')

    return sel_idx, samples_per_cls_cur, loss_mem


def main():
    # args & cfg
    args = parse_args()
    cfg = get_cfg(args)
    cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # write cfg
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.cfg')
    with open(log_file, 'a') as f:
        f.write(cfg.pretty_text)

    # logger
    logger = build_logger(cfg.work_dir, cfgname=cfg.cfgname)
    writer = SummaryWriter(log_dir=os.path.join(cfg.work_dir, f'tensorboard'))

    '''
    # -----------------------------------------
    # build model & optimizer
    # -----------------------------------------
    '''
    # build source model & load weights
    src_model = build_model(cfg.src_model)
    src_model = src_model.cuda()

    print(f'==> Loading checkpoint "{cfg.load}"')
    ckpt = torch.load(cfg.load, map_location='cuda')
    src_model.load_state_dict(ckpt['model_state'])

    # build target model
    model = set_model(cfg)

    optimizer = set_optimizer(model, cfg)

    test_criterion = build_loss(cfg.loss.test).cuda()
    print('==> Model built.')

    '''
    # -----------------------------------------
    # build dataset/dataloader
    # -----------------------------------------
    '''
    test_loader = build_divm_loader(cfg, mode='test')
    '''
    # -----------------------------------------
    # Test source model before distill 
    # -----------------------------------------
    '''
    # 如果要测量每个类别的精度，需要在配置文件(BETA_{}.py)中添加test_class_acc:True
    if cfg.get('test_class_acc', False):
        test_class_acc(test_loader, src_model, test_criterion, 0, logger, writer, cfg)
    else:
        test(test_loader, src_model, test_criterion, 0, logger, writer)

    '''
    # -----------------------------------------
    # Predict target 
    # -----------------------------------------
    '''
    # f_s预测的伪标签、真实标签、预测概率（最大分量不变，其余分量求和后取平均）
    tgt_psl, gt_labels, pred_mem = pred_target(test_loader, src_model, 0, logger, cfg)
    # 获取单独训练的样本索引，之后根据样本索引得到对应的dataloader
    num_sample = len(tgt_psl)
    loss_mem = torch.zeros(num_sample)
    fea_bank = torch.randn(num_sample, 256)
    score_bank = torch.randn(num_sample, cfg.num_classes).cuda()
    sel_idx, samples_per_cls, loss_mem = obtain_sel_samples_indices(test_loader, src_model, tgt_psl, loss_mem, 0, cfg, logger)
    warmup_loader_idx = build_divm_loader(cfg, mode='warmup', return_idx=True)  # 样本、真实标签(未使用)、索引
    if len(sel_idx) > 0:
        sel_loader_idx = build_divm_loader(cfg, mode='select', indices=sel_idx, psl=tgt_psl[sel_idx], return_idx=True)
    '''
    # -----------------------------------------
    # Start target training
    # -----------------------------------------
    '''
    print("==> Start training...")
    model.train()

    test_meter = TrackMeter()
    start_epoch = 1
    for epoch in range(start_epoch, cfg.epochs + 1):
        adjust_lr(optimizer, epoch, cfg.epochs, power=1.5)

        # momentum update pred_mem
        if epoch % cfg.pred_interval == 0:
            _, _, pred_t = pred_target(test_loader, model, epoch, logger, cfg)
            pred_mem = cfg.ema * pred_mem + (1 - cfg.ema) * pred_t
            tgt_psl = torch.max(pred_mem, dim=1).indices
            print(torch.mean((tgt_psl == gt_labels).float()))
            model.train()

        if epoch <= cfg.warmup_epochs:
            logger.info(f'==> Start distillation ...')
            dist_train(warmup_loader_idx, model, optimizer, epoch, logger, cfg, pred_mem)
        else:
            logger.info(f'==> Start contrast learning ...')
            contrast_learning(warmup_loader_idx, model, optimizer, epoch, logger, cfg, writer, fea_bank, score_bank)

        logger.info(f'==> Start select training ...')
        if len(sel_idx) > 0:
            select_train(sel_loader_idx, model, optimizer, epoch, logger, cfg, pred_mem)
        else:
            logger.info(f'Skip Selection training at epoch [{epoch}] - num_select: {len(sel_idx)}.')

        logger.info(f'==> Obtain select train loader ...')
        sel_idx, samples_per_cls, loss_mem = obtain_sel_samples_indices(test_loader, model, tgt_psl, loss_mem,
                                                                       epoch, cfg, logger)
        if len(sel_idx) > 0:
            sel_loader_idx = build_divm_loader(cfg, mode='select', indices=sel_idx, psl=tgt_psl[sel_idx], return_idx=True)

        if epoch % cfg.test_interval == 0 or epoch == cfg.epochs:
            if cfg.get('test_class_acc', False):
                test_acc, mean_ent, pred_max = \
                    test_class_acc(test_loader, model, test_criterion, epoch, logger, writer, cfg)
            else:
                test_acc, mean_ent = test(test_loader, model, test_criterion, epoch, logger, writer)
            test_meter.update(test_acc, idx=epoch)

    # We print the best test_acc but use the last checkpoint for fine-tuning.
    logger.info(f'Best test_Acc@1: {test_meter.max_val:.2f} (epoch={test_meter.max_idx}).')

    # save last
    model_path = os.path.join(cfg.work_dir, 'last.pth')
    state_dict = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epochs': cfg.epochs
    }
    torch.save(state_dict, model_path)


if __name__ == '__main__':
    main()
