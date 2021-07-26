import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from configs.default import get_config
from data import build_dataloader
from models import build_model
from losses import build_losses
from tools.eval_metrics import evaluate
from tools.utils import AverageMeter, Logger, save_checkpoint, set_seed


def parse_option():
    parser = argparse.ArgumentParser(description='Train image-based re-id model')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory")
    parser.add_argument('--dataset', type=str, help="market1501, cuhk03, dukemtmcreid, msmt17")
    # Miscs
    parser.add_argument('--output', type=str, help="your output path to save model and logs")
    parser.add_argument('--eval', type=int, default=0, help="evaluation only")
    parser.add_argument('--resume', type=str, metavar='PATH')    
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    # for lce
    parser.add_argument('--train_half', type=int, default=0, help='train with the first half of the datasets')    
    parser.add_argument('--save_lcefeat', default=0, type=int, help="if save the feats on the training datasets")
    parser.add_argument('--use_lce', type=int, default=0, help='if use lce')
    parser.add_argument('--use_trans', type=int, default=0, help='if use trans')    
    parser.add_argument('--path_ccb', type=str, help="path to the class centers and boundaries")
    parser.add_argument('--lambda_a', type=float, default=100, help="weight for align loss")
    parser.add_argument('--lambda_b', type=float, default=1, help="weight for boundary loss")    

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return config


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU

    if not config.EVAL_MODE:
        sys.stdout = Logger(osp.join(config.OUTPUT, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(config.OUTPUT, 'log_test.txt'))
    print("==========\nConfig:{}\n==========".format(config))
    print("Currently using GPU {}".format(config.GPU))
    # Set random seed
    set_seed(config.SEED)

    # Build dataloader
    trainloader, queryloader, galleryloader, num_classes = build_dataloader(config)
    # Build model
    if config.LCE.USE_TRANS:
        model, classifier, trans_forward, trans_backward = build_model(config, num_classes)
    else:
        model, classifier = build_model(config, num_classes)
    # Build classification and pairwise loss
    criterion_cla = build_losses(config)
    # Build optimizer
    parameters = list(model.parameters()) + list(classifier.parameters())
    if config.TRAIN.OPTIMIZER.NAME == 'adam':
        optimizer = optim.Adam(parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                               weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.NAME == 'adamw':
        optimizer = optim.AdamW(parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                               weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.NAME == 'sgd':
        optimizer = optim.SGD(parameters, lr=config.TRAIN.OPTIMIZER.LR, momentum=0.9, 
                              weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY, nesterov=True)
    else:
        raise KeyError("Unknown optimizer: {}".format(config.TRAIN.OPTIMIZER.NAME))
    # Build lr_scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.TRAIN.LR_SCHEDULER.STEPSIZE, 
                                         gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE)

    start_epoch = config.TRAIN.START_EPOCH
    if config.MODEL.RESUME:
        print("Loading checkpoint from '{}'".format(config.MODEL.RESUME))
        checkpoint = torch.load(config.MODEL.RESUME)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    model = nn.DataParallel(model).cuda()
    classifier = nn.DataParallel(classifier).cuda()
    if config.LCE.USE_TRANS:
        trans_forward = nn.DataParallel(trans_forward).cuda()
        trans_backward = nn.DataParallel(trans_backward).cuda()
    else:
        trans_forward, trans_backward = None, None
    
    if config.EVAL_MODE:
        print("Evaluate only")
        if config.LCE.SAVE_LCEFEAT:
            save_lcefeat(model, trainloader, config.OUTPUT)
            return
        test(model, queryloader, galleryloader, config.OUTPUT)
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")

    # load LCE files
    old_class_centers, old_class_cos = None, None
    lambda_a, lambda_b = 0, 0
    if config.LCE.USE_LCE:
        # note here old class centers are normalized
        old_class_centers = np.load('{}/old_class_centers.npy'.format(config.LCE.PATH_CCB))
        old_class_cos = np.load('{}/old_class_cos.npy'.format(config.LCE.PATH_CCB))
        old_class_centers = torch.Tensor(old_class_centers).cuda()
        old_class_centers = F.normalize(old_class_centers, p=2, dim=1)
        old_class_cos = torch.Tensor(old_class_cos).cuda()
        lambda_a = config.LCE.LAMBDA_A 
        lambda_b = config.LCE.LAMBDA_B

    for epoch in range(start_epoch, config.TRAIN.MAX_EPOCH):
        start_train_time = time.time()
        train(epoch, model, classifier, criterion_cla, optimizer, trainloader, use_lce=config.LCE.USE_LCE, old_class_centers=old_class_centers, old_class_cos=old_class_cos, lambda_a=lambda_a, lambda_b=lambda_b, use_trans=config.LCE.USE_TRANS, trans_forward=trans_forward, trans_backward=trans_backward)
        train_time += round(time.time() - start_train_time)        
        
        if (epoch+1) > config.TEST.START_EVAL and config.TEST.EVAL_STEP > 0 and \
            (epoch+1) % config.TEST.EVAL_STEP == 0 or (epoch+1) == config.TRAIN.MAX_EPOCH:
            print("==> Test")
            rank1 = test(model, queryloader, galleryloader)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            state_dict = model.module.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(config.OUTPUT, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))
            if config.LCE.USE_TRANS:
                state_dict_transf = trans_forward.module.state_dict()
                save_checkpoint({
                    'state_dict': state_dict_transf,
                    'rank1': rank1,
                    'epoch': epoch,
                }, is_best, osp.join(config.OUTPUT, 'trans_f' + str(epoch+1) + '.pth.tar'))
                state_dict_transb = trans_backward.module.state_dict()
                save_checkpoint({
                    'state_dict': state_dict_transb,
                    'rank1': rank1,
                    'epoch': epoch,
                }, is_best, osp.join(config.OUTPUT, 'trans_b' + str(epoch+1) + '.pth.tar'))                

        scheduler.step()

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


def train(epoch, model, classifier, criterion_cla, optimizer, trainloader, use_lce=False, old_class_centers=None, old_class_cos=None, lambda_a=0, lambda_b=0, use_trans=False, trans_forward=None, trans_backward=None):
    batch_cla_loss = AverageMeter()
    # batch_pair_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    if use_lce:
        batch_align_loss = AverageMeter()
        batch_bound_loss = AverageMeter()

    model.train()
    classifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        imgs, pids = imgs.cuda(), pids.cuda()
        # Measure data loading time
        data_time.update(time.time() - end)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward
        features = model(imgs)
        outputs = classifier(features)
        _, preds = torch.max(outputs.data, 1)
        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        if use_lce:
            # align loss
            if use_trans:
                align_loss = (torch.mean(F.mse_loss(F.normalize(classifier.module.weight), trans_forward(old_class_centers))) + torch.mean(F.mse_loss(trans_backward(F.normalize(classifier.module.weight)), old_class_centers)))/2
                cos_theta = torch.mm(trans_backward(features), old_class_centers.t())
            else:
                align_loss = torch.mean(F.mse_loss(F.normalize(classifier.module.weight), old_class_centers))  
                cos_theta = torch.mm(features, old_class_centers.t()).clamp(-1, 1)                    
            index = outputs.data * 0.0 #size=(B,Classnum)
            index.scatter_(1,pids.data.view(-1,1),1)
            index = index.byte().bool()
            val_all = torch.sum(index * cos_theta, dim=1)
            bound_loss = torch.mean((old_class_cos[pids] - val_all).clamp(0))            
        loss = cla_loss + lambda_a * align_loss + lambda_b * bound_loss
        # Backward + Optimize        
        loss.backward()
        optimizer.step()
        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))        
        # measure elapsed time
        if use_lce:
            # 1e5 for display
            batch_align_loss.update(align_loss.item()*1e5, pids.size(0))
            batch_bound_loss.update(bound_loss.item()*1e5, pids.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
    if use_lce:
        print('Epoch{0} '
            'Time:{batch_time.sum:.1f}s '
            'Data:{data_time.sum:.1f}s '
            'AlignLoss:{align_loss.avg:.4f} '
            'BoundLoss:{bound_loss.avg:.4f} '            
            'ClaLoss:{cla_loss.avg:.4f} '
            'Acc:{acc.avg:.2%} '.format(
            epoch+1, batch_time=batch_time, data_time=data_time, 
            align_loss=batch_align_loss, bound_loss=batch_bound_loss,
            cla_loss=batch_cla_loss, acc=corrects))        
    else:
        print('Epoch{0} '
            'Time:{batch_time.sum:.1f}s '
            'Data:{data_time.sum:.1f}s '
            'ClaLoss:{cla_loss.avg:.4f} '
            'Acc:{acc.avg:.2%} '.format(
            epoch+1, batch_time=batch_time, data_time=data_time, 
            cla_loss=batch_cla_loss, acc=corrects))


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)

    return img_flip


@torch.no_grad()
def extract_feature(model, dataloader):
    features, pids, camids = [], [], []
    for batch_idx, (imgs, batch_pids, batch_camids) in enumerate(dataloader):
        flip_imgs = fliplr(imgs)
        imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
        batch_features = model(imgs).data.cpu()
        batch_features_flip = model(flip_imgs).data.cpu()
        batch_features += batch_features_flip

        features.append(batch_features)
        pids.append(batch_pids)
        camids.append(batch_camids)
    features = torch.cat(features, 0)
    pids = torch.cat(pids, 0).numpy()
    camids = torch.cat(camids, 0).numpy()

    return features, pids, camids


def test(model, queryloader, galleryloader, save_dir=None):
    since = time.time()
    model.eval()
    # Extract features for query set
    qf, q_pids, q_camids = extract_feature(model, queryloader)
    print("Extracted features for query set, obtained {} matrix".format(qf.shape))
    # Extract features for gallery set
    gf, g_pids, g_camids = extract_feature(model, galleryloader)
    print("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    time_elapsed = time.time() - since
    print('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m,n))
    if config.TEST.DISTANCE == 'euclidean':
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        for i in range(m):
            distmat[i:i+1].addmm_(1, -2, qf[i:i+1], gf.t())
    else:
        # Cosine similarity
        qf = F.normalize(qf, p=2, dim=1)
        gf = F.normalize(gf, p=2, dim=1)
        for i in range(m):
            distmat[i] = - torch.mm(qf[i:i+1], gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    
    if save_dir is not None:
        np.savez('{}/query.npz'.format(save_dir), qf, q_pids, q_camids)
        np.savez('{}/gallery.npz'.format(save_dir), gf, g_pids, g_camids)

    print("Results ----------------------------------------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], mAP))
    print("------------------------------------------------")

    return cmc[0]


def save_lcefeat(model, trainloader, save_dir):
    since = time.time()
    model.eval()
    # Extract features for train set
    tf, t_pids, t_camids = extract_feature(model, trainloader)
    print("Extracted features for train set, obtained {} matrix".format(tf.shape))
    time_elapsed = time.time() - since
    print('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    tf = F.normalize(tf, p=2, dim=1)
    
    pid_features = {}
    for i, pid in enumerate(t_pids):
        pid_features.setdefault(pid, []).append(tf[i].cpu().numpy())

    # generate pid class centers
    old_class_centers = []
    old_class_cos = []
    pid_list = sorted(list(set(t_pids)))
    for pid in pid_list:
        local_features = pid_features[pid]
        # class center
        local_class_center = np.mean(np.array(local_features), axis=0)
        local_class_center = local_class_center/np.linalg.norm(local_class_center)
        old_class_centers.append(local_class_center)
        # cos values
        old_class_cos.append(min(np.dot(local_features, local_class_center)))
    
    # save
    np.save('{}/old_class_centers.npy'.format(save_dir), np.array(old_class_centers))
    np.save('{}/old_class_cos.npy'.format(save_dir), np.array(old_class_cos))
    return   


if __name__ == '__main__':
    config = parse_option()
    main(config)