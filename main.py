import os
import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from configs.default_vid import get_vid_config
from tools.utils import set_seed, get_logger, AverageMeter, save_checkpoint
from data import build_dataloader, VID_DATASET
from losses import build_losses
from models.vid_resnet import ResNet503D
from tools.eval_metrics import evaluate, evaluate_with_clothes


class CosineClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.normal_(self.weight, std=0.01)
    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        return F.linear(x, w)


def init_distributed(backend):
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", "0"))
    local_rank_env = os.environ.get("LOCAL_RANK")
    if local_rank_env is not None:
        local_rank = int(local_rank_env)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
        return dist.get_rank(), dist.get_world_size(), local_rank
    rank, world_size = 0, 1
    init_method = "tcp://127.0.0.1:29500"
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method=init_method, rank=rank, world_size=world_size)
    return rank, world_size, 0


def build_config(args):
    class ArgsForConfig:
        def __init__(self, ns):
            self.cfg = ns.cfg
            self.root = ns.root
            self.output = ns.output
            self.resume = ns.resume
            self.eval = ns.eval
            self.tag = ns.tag
            self.dataset = ns.dataset
            self.gpu = ns.gpu
            self.amp = ns.amp
    cfg_args = ArgsForConfig(args)
    config = get_vid_config(cfg_args)
    if args.dataset is not None:
        config.defrost(); config.DATA.DATASET = args.dataset; config.freeze()
    if config.DATA.DATASET not in VID_DATASET:
        config.defrost(); config.DATA.DATASET = "ccvid"; config.freeze()
    return config


def build_model_and_heads(config, dataset, backbone_path=None):
    num_frames = config.AUG.SEQ_LEN
    if backbone_path is not None:
        model = ResNet503D(num_frames=num_frames, backbone_path=backbone_path)
    else:
        model = ResNet503D(num_frames=num_frames)
    feat_dim = config.MODEL.FEATURE_DIM
    id_classifier = CosineClassifier(feat_dim, dataset.num_train_pids)
    clothes_classifier = CosineClassifier(feat_dim, dataset.num_train_clothes)
    return model, id_classifier, clothes_classifier


def compute_positive_mask(pids, dataset):
    device = pids.device
    num_clothes = dataset.num_train_clothes
    pid2clothes = torch.tensor(dataset.pid2clothes, device=device, dtype=torch.float32)
    mask = pid2clothes[pids.long()]
    mask = F.pad(mask, (0, num_clothes - mask.size(1))) if mask.size(1) != num_clothes else mask
    return mask


def train_one_epoch(model, id_head, clothes_head, criterion_cla, criterion_pair, criterion_clothes, criterion_cal, optimizer, scaler, trainloader, epoch, config, dataset, logger, local_rank):
    model.train()
    id_head.train()
    clothes_head.train()
    loss_meter = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    start = time.time()
    for batch_idx, batch in enumerate(trainloader):
        data_time.update(time.time() - start)
        clips, pids, camids, clothes_ids = batch
        clips = clips.cuda(non_blocking=True)
        pids = pids.cuda(non_blocking=True)
        clothes_ids = clothes_ids.cuda(non_blocking=True)
        with autocast(enabled=config.TRAIN.AMP):
            final_feat, a_feats, model_loss = model(clips, pids, True, epoch, batch_idx)
            id_logits = id_head(final_feat)
            a_mean = a_feats.mean(1)
            clothes_logits = clothes_head(a_mean)
            loss_id = criterion_cla(id_logits, pids)
            loss_pair = criterion_pair(final_feat, pids) * float(config.LOSS.PAIR_LOSS_WEIGHT)
            pos_mask = compute_positive_mask(pids, dataset)
            loss_cal = criterion_cal(clothes_logits, clothes_ids, pos_mask)
            loss_clothes = criterion_clothes(clothes_logits, clothes_ids)
            loss = model_loss + loss_id + loss_pair + loss_clothes + loss_cal
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_meter.update(loss.item(), clips.size(0))
        batch_time.update(time.time() - start)
        start = time.time()
        if local_rank in [-1, 0] and batch_idx % 10 == 0:
            logger.info("Epoch {} [{}/{}]  Loss {:.4f} ({:.4f})  Data {:.3f}s  Batch {:.3f}s".format(epoch, batch_idx, len(trainloader), loss_meter.val, loss_meter.avg, data_time.val, batch_time.val))
    return loss_meter.avg


def concat_all_gather(tensor):
    tensors_gather = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0)


def infer_features(model, loader, config):
    model.eval()
    feats = []
    pids_all = []
    camids_all = []
    clothes_all = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            clips, pids, camids, clothes_ids = batch
            clips = clips.cuda(non_blocking=True)
            pids = pids.cuda(non_blocking=True)
            f = model(clips, pids, False, 0, batch_idx)
            feats.append(f)
            pids_all.append(pids)
            camids_all.append(camids.cuda(non_blocking=True))
            clothes_all.append(clothes_ids.cuda(non_blocking=True))
    feats = torch.cat(feats, dim=0)
    pids_all = torch.cat(pids_all, dim=0)
    camids_all = torch.cat(camids_all, dim=0)
    clothes_all = torch.cat(clothes_all, dim=0)
    feats = concat_all_gather(feats)
    pids_all = concat_all_gather(pids_all)
    camids_all = concat_all_gather(camids_all)
    clothes_all = concat_all_gather(clothes_all)
    return feats, pids_all, camids_all, clothes_all


def aggregate_video_features(clip_feats, vid2clip_index):
    video_feats = []
    for indices in vid2clip_index:
        if len(indices) == 0:
            video_feats.append(torch.zeros_like(clip_feats[0]))
        else:
            video_feats.append(clip_feats[indices].mean(dim=0))
    return torch.stack(video_feats, dim=0)


def evaluate_reid(model, queryloader, galleryloader, dataset, logger, local_rank):
    q_feats, q_pids, q_camids, q_clothids = infer_features(model, queryloader, None)
    g_feats, g_pids, g_camids, g_clothids = infer_features(model, galleryloader, None)
    if local_rank not in [-1, 0]:
        return None
    with torch.no_grad():
        q_video_feats = aggregate_video_features(q_feats, dataset.query_vid2clip_index)
        g_video_feats = aggregate_video_features(g_feats, dataset.gallery_vid2clip_index)
        q_video_feats = F.normalize(q_video_feats, p=2, dim=1)
        g_video_feats = F.normalize(g_video_feats, p=2, dim=1)
        distmat = torch.cdist(q_video_feats, g_video_feats, p=2).cpu().numpy()
        q_pids_np = q_pids.cpu().numpy()
        g_pids_np = g_pids.cpu().numpy()
        q_camids_np = q_camids.cpu().numpy()
        g_camids_np = g_camids.cpu().numpy()
        q_cloth_np = q_clothids.cpu().numpy()
        g_cloth_np = g_clothids.cpu().numpy()
        CMC, mAP = evaluate(distmat, q_pids_np, g_pids_np, q_camids_np, g_camids_np)
        CMC_CC, mAP_CC = evaluate_with_clothes(distmat, q_pids_np, g_pids_np, q_camids_np, g_camids_np, q_cloth_np, g_cloth_np, mode="CC")
        CMC_SC, mAP_SC = evaluate_with_clothes(distmat, q_pids_np, g_pids_np, q_camids_np, g_camids_np, q_cloth_np, g_cloth_np, mode="SC")
        logger.info("Eval Results: R1 {:.2%} mAP {:.2%} | CC R1 {:.2%} mAP {:.2%} | SC R1 {:.2%} mAP {:.2%}".format(CMC[0], mAP, CMC_CC[0], mAP_CC, CMC_SC[0], mAP_SC))
        return dict(R1=float(CMC[0]), mAP=float(mAP), R1_CC=float(CMC_CC[0]), mAP_CC=float(mAP_CC), R1_SC=float(CMC_SC[0]), mAP_SC=float(mAP_SC))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--backbone_path", type=str, default=None)
    args = parser.parse_args()
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    rank, world_size, local_rank = init_distributed(backend)
    config = build_config(args)
    if args.amp:
        config.defrost(); config.TRAIN.AMP = True; config.freeze()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(len(config.GPU.split(",")))]) if config.GPU else ""
    set_seed(config.SEED)
    log_file = os.path.join(config.OUTPUT, "train_eval.log")
    logger = get_logger(log_file, local_rank=rank, name="hamobe")
    trainloader, queryloader, galleryloader, dataset, train_sampler = build_dataloader(config)
    model, id_head, clothes_head = build_model_and_heads(config, dataset, args.backbone_path)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    id_head = id_head.to(device)
    clothes_head = clothes_head.to(device)
    model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=True)
    id_head = DDP(id_head, device_ids=[local_rank] if torch.cuda.is_available() else None)
    clothes_head = DDP(clothes_head, device_ids=[local_rank] if torch.cuda.is_available() else None)
    criterion_cla, criterion_pair, criterion_clothes, criterion_cal = build_losses(config, dataset.num_train_clothes)
    criterion_cla = criterion_cla.to(device)
    criterion_pair = criterion_pair.to(device)
    criterion_clothes = criterion_clothes.to(device)
    criterion_cal = criterion_cal.to(device)
    params = list(model.parameters()) + list(id_head.parameters()) + list(clothes_head.parameters())
    optimizer = torch.optim.Adam(params, lr=float(config.TRAIN.OPTIMIZER.LR), weight_decay=float(config.TRAIN.OPTIMIZER.WEIGHT_DECAY))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(config.TRAIN.LR_SCHEDULER.STEPSIZE), gamma=float(config.TRAIN.LR_SCHEDULER.DECAY_RATE))
    scaler = GradScaler(enabled=config.TRAIN.AMP)
    best_map = 0.0
    start_epoch = int(config.TRAIN.START_EPOCH)
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.module.load_state_dict(ckpt.get("model", {}), strict=False)
        id_head.module.load_state_dict(ckpt.get("id_head", {}), strict=False)
        clothes_head.module.load_state_dict(ckpt.get("clothes_head", {}), strict=False)
        optimizer.load_state_dict(ckpt.get("optimizer", optimizer.state_dict()))
        scheduler.load_state_dict(ckpt.get("scheduler", scheduler.state_dict()))
        start_epoch = ckpt.get("epoch", start_epoch)
        best_map = ckpt.get("best_map", best_map)
        logger.info("Resumed from {}".format(args.resume))
    if args.eval or bool(config.EVAL_MODE):
        eval_res = evaluate_reid(model, queryloader, galleryloader, dataset, logger, local_rank)
        if rank in [0]:
            logger.info("Evaluation done.")
        return
    max_epoch = int(config.TRAIN.MAX_EPOCH)
    for epoch in range(start_epoch, max_epoch):
        train_sampler.set_epoch(epoch)
        loss_avg = train_one_epoch(model, id_head, clothes_head, criterion_cla, criterion_pair, criterion_clothes, criterion_cal, optimizer, scaler, trainloader, epoch, config, dataset, logger, local_rank)
        scheduler.step()
        do_eval = (int(config.TEST.EVAL_STEP) > 0 and (epoch + 1) % int(config.TEST.EVAL_STEP) == 0 and (epoch + 1) >= int(config.TEST.START_EVAL))
        if do_eval:
            eval_res = evaluate_reid(model, queryloader, galleryloader, dataset, logger, local_rank)
            if rank in [0] and eval_res is not None:
                current_map = eval_res["mAP"]
                is_best = current_map > best_map
                best_map = max(best_map, current_map)
                state = dict(model=model.module.state_dict(), id_head=id_head.module.state_dict(), clothes_head=clothes_head.module.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), epoch=epoch + 1, best_map=best_map)
                ckpt_path = os.path.join(config.OUTPUT, "checkpoint.pth.tar")
                save_checkpoint(state, is_best, ckpt_path)
                logger.info("Epoch {} done. Loss {:.4f}. Best mAP {:.2%}".format(epoch + 1, loss_avg, best_map))
    if rank in [0]:
        logger.info("Training finished.")


if __name__ == "__main__":
    main()
