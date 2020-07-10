import os
import fire
import ttach
import torch
import addict
import argparse
import rasterio
import pandas as pd
import cv2

from tqdm import tqdm
from multiprocessing import Pool

from rasterio.windows import Window

from .training.runner import GPUNormRunner
from .training.config import parse_config

from . import getters
from .training.predictor import TorchTifPredictor
from .datasets import TestSegmentationDataset


class EnsembleModel(torch.nn.Module):
    """Ensemble of torch models, pass tensor through all models and average results"""

    def __init__(self, models: list):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        result = None
        for model in self.models:
            y = model(x)
            if result is None:
                result = y
            else:
                result += y
        result /= torch.tensor(len(self.models)).to(result.device)
        return result


def model_from_config(path: str):
    """Create model from configuration specified in config file and load checkpoint weights"""
    cfg = addict.Dict(parse_config(config=path))  # read and parse config file
    init_params = cfg.model.init_params  # extract model initialization parameters
    init_params["encoder_weights"] = None  # because we will load pretrained weights for whole model
    model = getters.get_model(architecture=cfg.model.architecture, init_params=init_params)
    checkpoint_path = os.path.join(cfg.logdir, "checkpoints", "best.pth")
    state_dict = torch.load(checkpoint_path)["state_dict"]
    model.load_state_dict(state_dict)
    return model


def read_tile(src_path, x, y, size):
    """Read square tile from big tif file according to specified coordinates (x, y) and spatial size"""
    with rasterio.open(src_path) as f:
        return f.read(window=Window(x, y, size, size)), f.profile


def write_tile(dst_path, tile, profile, **kwargs):
    """Save tile to disk"""
    # save tile with shape 1024x1024 because this is initial size of test images
    profile.update(dict(height=1024, width=1024, NBITS=1, count=1))
    profile.update(**kwargs)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(tile)


def slice_to_tiles(args):
    """Slice big tif file to small tiles"""
    src_dir, dst_dir, row = args
    src_path = os.path.join(src_dir, str(row.cluster_id).zfill(3) + '.tif')
    tile, profile = read_tile(src_path, row.x * row.tile_size, row.y * row.tile_size, row.tile_size)
    dst_path = os.path.join(dst_dir, row.id)
    write_tile(dst_path, tile, profile)


def main(args):
    # set GPUS
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # define where stithced and not stitched test data is located
    src_sliced_dir = os.path.join(args.test_dir, "images")

    # prepare output (prediction) dirs
    dst_sliced_dir = os.path.join(args.dst_dir, "images")

    os.makedirs(dst_sliced_dir, exist_ok=True)

    # --------------------------------------------------
    # define model
    # --------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Available devices:", device)

    # loading trained models
    models = [model_from_config(config_path) for config_path in args.configs]

    # create ensemble
    model = EnsembleModel(models)

    # add test time augmentations (flipping and rotating input image)
    model = ttach.SegmentationTTAWrapper(model, ttach.aliases.d4_transform(), merge_mode='mean')

    # create Multi GPU model if number of GPUs is more than one
    n_gpus = len(args.gpu.split(","))
    if n_gpus > 1:
        gpus = list(range(n_gpus))
        model = torch.nn.DataParallel(model, gpus)

    print("Done loading...")
    model.to(device)

    # --------------------------------------------------
    # start evaluation
    # --------------------------------------------------
    runner = GPUNormRunner(model, model_device=device)
    model.eval()

    # predict not stitched data
    print("Predicting initial files")
    test_dataset = TestSegmentationDataset(src_sliced_dir, transform_name='test_transform_1')
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=8,
    )

    for batch in tqdm(test_dataloader):
        ids = batch['id']
        predictions = runner.predict_on_batch(batch)['mask']

        for image_id, prediction in zip(ids, predictions):
            prediction = prediction.round().int().cpu().numpy().astype("uint8")
            profile = test_dataset.read_image_profile(image_id)
            dst_path = os.path.join(dst_sliced_dir, image_id)
            cv2.imwrite(dst_path, prediction)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs="+", required=True, type=str)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--dst_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    main(args)
    os._exit(0)