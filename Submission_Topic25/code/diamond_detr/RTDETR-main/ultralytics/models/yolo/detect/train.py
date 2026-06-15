# Minimal DetectionTrainer compatible with fork ultralytics 8.0.201.
# Only implements what RTDETRTrainer actually inherits.
# Avoids API mismatches with stock ultralytics 8.4.64.

from copy import copy

import torch

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import torch_distributed_zero_first, de_parallel


class DetectionTrainer(BaseTrainer):

    def build_dataset(self, img_path, mode='train', batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        assert mode in ('train', 'val')
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == 'train'
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle, rank)

    def preprocess_batch(self, batch):
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        return batch

    def set_model_attributes(self):
        self.model.nc = self.data['nc']
        self.model.names = self.data['names']
        self.model.args = self.args

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        from ultralytics.models.yolo.detect.val import DetectionValidator
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return DetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def label_loss_items(self, loss_items=None, prefix='train'):
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        return keys

    def progress_string(self):
        return ('\n' + '%11s' * (4 + len(self.loss_names))) % (
            'Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def plot_training_samples(self, batch, ni):
        cls = batch['cls'].squeeze(-1)
        if cls.ndim == 0:
            cls = cls.unsqueeze(0)
        try:
            plot_images(images=batch['img'],
                        batch_idx=batch['batch_idx'],
                        cls=cls,
                        bboxes=batch['bboxes'],
                        paths=batch['im_file'],
                        fname=self.save_dir / f'train_batch{ni}.jpg',
                        on_plot=self.on_plot)
        except Exception:
            pass

    def plot_training_labels(self):
        boxes = self.train_loader.dataset.labels
        if not boxes:
            return
        try:
            b = [lb['bboxes'] for lb in boxes]
            c = [lb['cls'] for lb in boxes]
            plot_labels(torch.cat(b, 0), torch.cat(c, 0).squeeze(), self.data['names'], self.save_dir,
                        on_plot=self.on_plot)
        except Exception:
            pass
