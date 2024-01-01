#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch

from einops import repeat
from iopath.common.file_io import g_pathmgr

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestMeter

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    if cfg.TEST.FEAT_EXTRACT:
        features_path = os.path.join(cfg.OUTPUT_DIR, 'features')
        try:
            if not os.path.exists(features_path):
                os.makedirs(features_path)
        except Exception as e:
            print(e)
        features_dict = {}

    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if key == 'vid':
                    continue
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()

        if cfg.TEST.FEAT_EXTRACT:
            if cfg.DETECTION.ENABLE:
                # Compute the predictions.
                preds = model(inputs, meta["boxes"])
                ori_boxes = meta["ori_boxes"]
                metadata = meta["metadata"]

                preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
                ori_boxes = (
                    ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
                )
                metadata = (
                    metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
                )

                if cfg.NUM_GPUS > 1:
                    preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                    ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                    metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            else:
                # Perform the forward pass.
                preds = model(inputs, feat_extract=True)
                # print(preds.shape)
                # print(meta['vid'])
                for pred_, vid_ in zip(preds, meta['vid']):
                    # _ = features_dict.setdefault(vid_, pred_.detach().cpu())
                    with open(os.path.join(cfg.OUTPUT_DIR, 'features', vid_.replace('.mp4', '.pkl')), 'wb') as f:
                        pickle.dump(pred_.detach().cpu(), f)

                # Gather all the predictions across all the devices to perform ensemble.
                # if cfg.NUM_GPUS > 1:
                #     preds, labels, video_idx = du.all_gather(
                #         [preds, labels, video_idx]
                #     )
                # if cfg.NUM_GPUS:
                #     preds = preds.cpu()
                #     labels = labels.cpu()
                #     video_idx = video_idx.cpu()

        else:
            if cfg.DETECTION.ENABLE:
                # Compute the predictions.
                preds = model(inputs, meta["boxes"])
                ori_boxes = meta["ori_boxes"]
                metadata = meta["metadata"]

                preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
                ori_boxes = (
                    ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
                )
                metadata = (
                    metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
                )

                if cfg.NUM_GPUS > 1:
                    preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                    ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                    metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

                test_meter.iter_toc()
                # Update and log stats.
                test_meter.update_stats(preds, ori_boxes, metadata)
                test_meter.log_iter_stats(None, cur_iter)
            else:
                # Perform the forward pass.
                preds = model(inputs)

                # Gather all the predictions across all the devices to perform ensemble.
                if cfg.NUM_GPUS > 1:
                    preds, labels, video_idx = du.all_gather(
                        [preds, labels, video_idx]
                    )
                if cfg.NUM_GPUS:
                    preds = preds.cpu()
                    labels = labels.cpu()
                    video_idx = video_idx.cpu()

                test_meter.iter_toc()
                # Update and log stats.
                test_meter.update_stats(
                    preds.detach(), labels.detach(), video_idx.detach()
                )
                test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    if cfg.TEST.FEAT_EXTRACT:
        # for vid_, feat_ in features_dict.items():
        #     with open(os.path.join(cfg.OUTPUT_DIR, 'features', vid_.replace('.mp4', '.pkl')), 'wb') as f:
        #         pickle.dump(feat_, f)

        return None

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            if du.is_root_proc():
                with g_pathmgr.open(save_path, "wb") as f:
                    pickle.dump([all_preds, all_labels], f)

            logger.info(
                "Successfully saved prediction results to {}".format(save_path)
            )

    test_meter.finalize_metrics()
    return test_meter

def test_one(model, test_meter, cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    if cfg.TEST.DENSE_SPATIAL_CROP:
        ratio = cfg.TEST.SPATIAL_SAMPLE_RATIO
        ratio_str = f"{ratio[0]:2f}-{ratio[1]:2f}"
        if cfg.TEST.CHECKPOINT_FILE_PATH != "":
            start_with = f"dense_crop_{int(cfg.TEST.CHECKPOINT_FILE_PATH[:-5].split('_')[-1])}e"
        else:
            start_with = "dense_crop"
        if 'from_train' in cfg.DATA.PM_SUBSET or 'split' in cfg.DATA.PM_SUBSET:
            trial_dir = f"{start_with}_{cfg.DATA.PM_SUBSET[1:]}_{cfg.DATA.TEST_CROP_SIZE}_min{cfg.DATA.TRAIN_JITTER_SCALES[0]}_{cfg.TEST.DENSE_SPATIAL_CROP_STEPS[0]}_{cfg.TEST.DENSE_SPATIAL_CROP_STEPS[1]}"
        elif '_val' in cfg.DATA.PM_SUBSET:
            trial_dir = f"{start_with}_test_val_{cfg.DATA.TEST_CROP_SIZE}_min{cfg.DATA.TRAIN_JITTER_SCALES[0]}_{cfg.TEST.DENSE_SPATIAL_CROP_STEPS[0]}_{cfg.TEST.DENSE_SPATIAL_CROP_STEPS[1]}"
        else:
            trial_dir = f"{start_with}_{cfg.DATA.TEST_CROP_SIZE}_min{cfg.DATA.TRAIN_JITTER_SCALES[0]}_{cfg.TEST.DENSE_SPATIAL_CROP_STEPS[0]}_{cfg.TEST.DENSE_SPATIAL_CROP_STEPS[1]}"
        try:
            if not os.path.exists(os.path.join(cfg.OUTPUT_DIR, trial_dir)): os.makedirs(os.path.join(cfg.OUTPUT_DIR, trial_dir))
        except Exception as e:
            logger.info(e)
        file_name = f'{trial_dir}/{cfg.DATA.NUM_FRAMES}x{cfg.DATA.TEST_CROP_SIZE}x{cfg.TEST.NUM_ENSEMBLE_VIEWS}x{cfg.TEST.NUM_SPATIAL_CROPS}_{ratio_str}.pkl'
        if os.path.exists(os.path.join(
                        cfg.OUTPUT_DIR, file_name)):
            logger.info(f"File {file_name} exists. Skip the test to next spatial position.")
            return None
    else:
        if cfg.TEST.PROCESS:
            if len(cfg.DATA.TEST_CROP_SIZE_RECT)!=0:
                crop_szie = f"{cfg.DATA.TEST_CROP_SIZE_RECT[0]}x{cfg.DATA.TEST_CROP_SIZE_RECT[1]}"
            else:
                crop_szie = f"{cfg.DATA.TEST_CROP_SIZE}x{cfg.DATA.TEST_CROP_SIZE}"
        else:
            if len(cfg.DATA.TRAIN_CROP_SIZE_RECT)!=0:
                crop_szie = f"{cfg.DATA.TRAIN_CROP_SIZE_RECT[0]}x{cfg.DATA.TRAIN_CROP_SIZE_RECT[1]}"
            else:
                crop_szie = f"{cfg.DATA.TRAIN_CROP_SIZE}x{cfg.DATA.TRAIN_CROP_SIZE}"
        file_name = f'{cfg.DATA.NUM_FRAMES}x{crop_szie}x{cfg.TEST.NUM_ENSEMBLE_VIEWS}x{cfg.TEST.NUM_SPATIAL_CROPS}_{cfg.TEST.SPATIAL_SAMPLE_INDEX}.pkl'

    # # Perform multi-view test on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
    if cfg.TEST.FEAT_EXTRACT:
        return None
    if writer is not None:
        writer.close()

    with g_pathmgr.open(os.path.join(
        cfg.OUTPUT_DIR, file_name),
        'wb'
    ) as f:
        result = {
            'video_preds': test_meter.video_preds.cpu().numpy(),
            'video_labels': test_meter.video_labels.cpu().numpy()
        }
        pickle.dump(result, f)

    return test_meter


def test(cfg):
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)
    
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (
            test_loader.dataset.num_videos
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )
        # Create meters for multi-view testing.
        test_meter = TestMeter(
            test_loader.dataset.num_videos
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.MODEL.NUM_CLASSES,
            len(test_loader),
            cfg.DATA.MULTI_LABEL,
            cfg.DATA.ENSEMBLE_METHOD,
        )

    if cfg.TEST.DENSE_SPATIAL_CROP:
        sample_step = cfg.TEST.DENSE_SPATIAL_CROP_STEPS
        h_sample_ratios = torch.linspace(0, 1, sample_step[0])
        w_sample_ratios = torch.linspace(0, 1, sample_step[1])
        merged_sample_ratios = torch.stack((repeat(h_sample_ratios, 
                                                    'h -> h n', 
                                                    n=len(w_sample_ratios)),
                                            repeat(w_sample_ratios, 
                                                    'w -> n w', 
                                                    n=len(h_sample_ratios))),
                                            dim=2).view(len(h_sample_ratios)*len(w_sample_ratios),
                                                        -1)
        for ratio in merged_sample_ratios:
            cfg.TEST.SPATIAL_SAMPLE_RATIO = ratio.tolist()
            logger.info(f"{ratio.tolist()[0]},{ratio.tolist()[1]}")
            test_one(model, test_meter, cfg)
            test_meter.reset_all()

        save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH, f"dense_crop_crop_{cfg.DATA.TEST_CROP_SIZE}_min{cfg.DATA.TRAIN_JITTER_SCALES[0]}_{cfg.TEST.DENSE_SPATIAL_CROP_STEPS[0]}_{cfg.TEST.DENSE_SPATIAL_CROP_STEPS[1]}.pkl")
        with open(save_path,'wb') as f:
            import pickle as pkl
            pkl.dump(test_meter.stats_dense_spatial, f)
    else:
        test_one(model, test_meter, cfg)