import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
t0 = time.time()
import argparse, os, sys, datetime, glob, importlib, csv, logging
import numpy as np
if not hasattr(np, "Inf"):
    np.Inf = np.inf
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
import torch
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
import torchvision
# import faulthandler
# faulthandler.enable()
t1 = time.time()
import pytorch_lightning as pl
t2 = time.time()
# print(f"import pytorch-lightning using {t2-t1} seconds")
import matplotlib.pyplot as plt

from tqdm import tqdm as _tqdm

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor

from pytorch_lightning.utilities import rank_zero_only, rank_zero_info

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
# from scripts.slice2seg import save_dice_hist
t3 = time.time()
# print(f"finish importing using {t3-t0} seconds")

# from pytorch_lightning.strategies import DDPStrategy




def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="outputs/logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--freeze_text",
        type=str2bool,
        nargs="?",
        const=True,
        default=None,
        help="override text_encoder_config.freeze (e.g. --freeze_text true)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="override data.params.batch_size",
    )
    parser.add_argument(
        "--eval_every_n_epochs",
        type=int,
        default=1,
        help="run IrisStyleLogger Dice/IoU evaluation every N epochs",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="dataset split for IrisStyleLogger Dice/IoU evaluation",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn, pin_memory=True)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle, pin_memory=True)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle, pin_memory=True)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            # ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            # trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=False, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=True,
                 log_images_kwargs=None, log_dice_frequency=None, enable_dice_log=False):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TensorBoardLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_dice_frequency = log_dice_frequency
        self.enable_dice_log = enable_dice_log

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    @rank_zero_only
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

        # log dice (disabled by default when IrisStyleLogger handles epoch-end evaluation)
        if (self.enable_dice_log and
                self.log_dice_frequency and
                check_idx % self.log_dice_frequency == 0 and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_dice") and
                callable(pl_module.log_dice) and
                self.max_images > 0 and
                check_idx > 0
        ):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                metrics_dict, seg_label_dict = pl_module.log_dice()
                dice_list = metrics_dict["val_avg_dice"]
                # print("[metric list]", metrics_dict["val_avg_dice"], metrics_dict["val_avg_iou"])
            for k in seg_label_dict:
                N = min(seg_label_dict[k].shape[0], self.max_images)
                seg_label_dict[k] = seg_label_dict[k][:N]
                if isinstance(seg_label_dict[k], torch.Tensor):
                    seg_label_dict[k] = seg_label_dict[k].detach().cpu()
                    if self.clamp:
                        seg_label_dict[k] = torch.clamp(seg_label_dict[k], -1., 1.)
            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, seg_label_dict, pl_module.global_step, split)

            for key, value in metrics_dict.items():
                if key == "val_avg_dice" or key == "val_avg_iou":
                    # print("[log metric 1]: ", key, value, sum(value), len(value), sum(value) / len(value))
                    # print(f"\033[31m###### {key}, {check_idx}, {pl_module.global_step}\033[0m")
                    pl_module.log(key, sum(value) / len(value),
                                  prog_bar=False, logger=True, on_step=True, on_epoch=False)
                elif "val_avg_dice" in key or "val_avg_iou" in key:     # isinstance(value, list) and len(value) > 0:
                    # print("[log metric 2]: ", key, value)
                    # print(f"\033[31m###### {key}, {check_idx}, {pl_module.global_step}\033[0m")
                    pl_module.log(key, value,
                                  prog_bar=False, logger=True, on_step=True, on_epoch=False)
                else:
                    continue

            # os.makedirs(os.path.join(pl_module.logger.save_dir, "dices_hist"), exist_ok=True)
            # save_dice_hist(
            #     dice_list,
            #     os.path.join(pl_module.logger.save_dir, "dices_hist"),
            #     dict(global_step=pl_module.global_step),
            # )
            # pl_module.logger.experiment.add_histogram("val_dice_hist", torch.tensor(dice_list),
            #                                           global_step=pl_module.global_step)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def _gpu(self, trainer):
        return getattr(trainer, "root_gpu", None) or getattr(trainer.strategy, "root_device", 0)

    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(self._gpu(trainer))
        torch.cuda.synchronize(self._gpu(trainer))
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs=None):
        torch.cuda.synchronize(self._gpu(trainer))
        max_memory = torch.cuda.max_memory_allocated(self._gpu(trainer)) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


class IrisStyleLogger(Callback):
    """Epoch-level console & file logger styled after iris/run.py."""

    def __init__(self, logdir: str, eval_split: str = "validation", eval_every_n_epochs: int = 1):
        super().__init__()
        self.logdir = logdir
        self.eval_split = eval_split
        self.eval_every_n_epochs = max(1, int(eval_every_n_epochs))
        self.logger = None
        self.best_dice = 0.0
        self.loss_sum = 0.0
        self.sample_count = 0
        self.epoch_start = None

    def _setup_logger(self):
        log_file = os.path.join(self.logdir, "train.log")
        os.makedirs(self.logdir, exist_ok=True)

        logger = logging.getLogger(f"tmfseg_{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers = []

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(asctime)s - %(message)s", "%H:%M:%S"))
        logger.addHandler(console)

        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

        return logger

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        self.logger = self._setup_logger()
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("TMF-Seg")
        self.logger.info(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 60)
        datasets = getattr(trainer.datamodule, "datasets", {})
        n_train = len(datasets["train"]) if "train" in datasets else 0
        n_val = len(datasets[self.eval_split]) if self.eval_split in datasets else 0
        self.logger.info(f"Train: {n_train} | Val ({self.eval_split}): {n_val}")
        self.logger.info("-" * 60)

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        self.loss_sum = 0.0
        self.sample_count = 0
        self.epoch_start = time.time()

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        loss_val = None
        if isinstance(outputs, dict) and "loss" in outputs:
            loss_val = outputs["loss"]
        elif isinstance(outputs, torch.Tensor):
            loss_val = outputs

        if loss_val is not None:
            if isinstance(loss_val, torch.Tensor):
                loss_val = loss_val.detach().item()
            bs = batch["image"].shape[0] if isinstance(batch, dict) and "image" in batch else 1
            self.loss_sum += float(loss_val) * bs
            self.sample_count += bs

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module, outputs=None):
        if self.logger is None:
            return

        epoch_time = time.time() - self.epoch_start if self.epoch_start is not None else 0.0
        avg_loss = self.loss_sum / max(self.sample_count, 1)
        epoch_num = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs if trainer.max_epochs and trainer.max_epochs > 0 else None
        epoch_str = f"Epoch [{epoch_num}/{max_epochs}]" if max_epochs else f"Epoch [{epoch_num}]"

        if (epoch_num % self.eval_every_n_epochs == 0) and hasattr(pl_module, "log_dice"):
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                metrics_dict, _ = pl_module.log_dice(split=self.eval_split, verbose=False)

            if is_train:
                pl_module.train()

            dice_list = metrics_dict["val_avg_dice"]
            iou_list = metrics_dict["val_avg_iou"]
            mean_dice = sum(dice_list) / len(dice_list)
            mean_iou = sum(iou_list) / len(iou_list)
            dice_pct = mean_dice * 100.0
            iou_pct = mean_iou * 100.0

            if dice_pct > self.best_dice:
                self.best_dice = dice_pct

            pl_module.log("val_avg_dice", mean_dice, prog_bar=False, logger=True, on_step=False, on_epoch=True, batch_size=1)
            pl_module.log("val_avg_iou", mean_iou, prog_bar=False, logger=True, on_step=False, on_epoch=True, batch_size=1)

            self.logger.info(
                f"{epoch_str} Loss: {avg_loss:.4f} | "
                f"Dice: {dice_pct:.2f}% | IoU: {iou_pct:.2f}% | "
                f"Time: {epoch_time:.1f}s"
            )
        else:
            self.logger.info(f"{epoch_str} Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")

    @rank_zero_only
    def on_fit_end(self, trainer, pl_module):
        if self.logger is not None:
            self.logger.info(f"Training done. Best Dice: {self.best_dice:.2f}%")


if __name__ == "__main__":
    # # set multiprocessing method to 'spawn'
    # # this can prevent: "RuntimeError: DataLoader worker (pid 227340) is killed by signal: Aborted."
    # # BUT: Don't change any code during training!
    # import multiprocessing
    # multiprocessing.set_start_method('spawn')   # TODO: remove this

    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)


    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        if opt.batch_size is not None:
            config.data.params.batch_size = opt.batch_size
        if opt.freeze_text is not None:
            cond_params = config.model.params.get("cond_stage_config", {}).get("params", {})
            if "text_encoder_config" in cond_params:
                cond_params.text_encoder_config.freeze = opt.freeze_text
        lightning_config = config.pop("lightning", OmegaConf.create())
        # config for datasets
        config.data.params.train.params.update({"num_classes": config.model.params.num_classes})
        config.data.params.validation.params.update({"num_classes": config.model.params.num_classes})
        config.data.params.test.params.update({"num_classes": config.model.params.num_classes})
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        trainer_config["accelerator"] = "gpu"   # "gpu" -> stable but lower; "ddp" -> unstable but higher
        # trainer_config["strategy"] = "ddp_find_unused_parameters_false"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TensorBoardLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                # "filename": "{epoch:06}-{step:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            if model.monitor == "val/rec_loss":
                default_modelckpt_cfg["params"]["monitor"] = model.monitor
                default_modelckpt_cfg["params"]["save_top_k"] = 1   # to save storage space
            elif model.monitor == "val_avg_dice":
                default_modelckpt_cfg["params"]["every_n_epochs"] = opt.eval_every_n_epochs
                default_modelckpt_cfg["params"]["save_on_train_epoch_end"] = True
                default_modelckpt_cfg["params"]["monitor"] = model.monitor
                default_modelckpt_cfg["params"]["save_top_k"] = 1   # to save storage space
                default_modelckpt_cfg["params"]["mode"] = "max"
            else:
                raise NotImplementedError(f"Not implemented for {model.monitor} monitor.")

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "main.CUDACallback"
            },
            "iris_logger": {
                "target": "main.IrisStyleLogger",
                "params": {
                    "logdir": logdir,
                    "eval_split": opt.eval_split,
                    "eval_every_n_epochs": opt.eval_every_n_epochs,
                }
            },
        }
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                     'params': {
                         "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                         "filename": "{epoch:06}-{step:09}",
                         "verbose": True,
                         'save_top_k': -1,
                         'every_n_train_steps': 5000,
                         'save_weights_only': True
                     }
                     }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        # trainer_kwargs["precision"] = 16    #
        # trainer_kwargs["strategy"] = "ddp_find_unused_parameters_false"
        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir  ###

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary, but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()
        print("#### Data #####")
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")


        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                # ckpt_path = os.path.join(ckptdir, "last.ckpt")
                # trainer.save_checkpoint(ckpt_path)


        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()


        import signal

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                print("end run")
                melk()
                print("end run")
                raise
        if not opt.no_test and not trainer.interrupted and trainer.global_rank == 0:
            if hasattr(model, "log_dice") and "test" in data.datasets:
                best_ckpt = getattr(trainer.checkpoint_callback, "best_model_path", None)
                if not best_ckpt:
                    best_ckpt = getattr(opt, "resume_from_checkpoint", None)
                if best_ckpt and os.path.exists(best_ckpt):
                    sd = torch.load(best_ckpt, map_location="cpu")
                    sd = sd.get("state_dict", sd)
                    model.load_state_dict(sd, strict=False)
                    print(f"Loaded best checkpoint for test: {best_ckpt}")
                if not cpu:
                    model = model.cuda()
                model.eval()
                with torch.no_grad():
                    test_loader = DataLoader(data.datasets["test"], batch_size=1, shuffle=False, pin_memory=True)
                    metrics_dict, _ = model.log_dice(data=test_loader, verbose=False)
                dice_list = metrics_dict["val_avg_dice"]
                iou_list = metrics_dict["val_avg_iou"]
                test_dice = (sum(dice_list) / len(dice_list)) * 100.0
                test_iou = (sum(iou_list) / len(iou_list)) * 100.0
                test_msg = f"Test | Dice: {test_dice:.2f}% | IoU: {test_iou:.2f}%"
                iris_cb = next((cb for cb in trainer.callbacks if isinstance(cb, IrisStyleLogger)), None)
                if iris_cb is not None and iris_cb.logger is not None:
                    iris_cb.logger.info(test_msg)
                else:
                    print(test_msg)

    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        print("end finally")

        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())
