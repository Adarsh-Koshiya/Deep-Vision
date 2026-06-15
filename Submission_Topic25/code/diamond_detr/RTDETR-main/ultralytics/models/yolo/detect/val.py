# Minimal DetectionValidator compatible with fork ultralytics 8.0.201.
# RTDETRValidator inherits from this and overrides postprocess/update_metrics.
# Uses the fork's DetMetrics/ConfusionMatrix APIs (different from stock 8.4.64).

import os
from pathlib import Path

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, colorstr, ops
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import plot_images
from ultralytics.utils.torch_utils import de_parallel


class DetectionValidator(BaseValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.numel()
        self.metrics = DetMetrics(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        self.args.task = 'detect'

    def preprocess(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        batch['img'] = (batch['img'].half() if self.args.half else batch['img'].float()) / 255
        return batch

    def init_metrics(self, model):
        self.names = model.names
        self.nc = len(model.names)
        self.seen = 0
        self.jdict = []
        self.stats = []
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.metrics.box = self.metrics.box.__class__()  # reset Metric
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf or 0.25)

    def get_desc(self):
        return ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95)')

    def postprocess(self, preds):
        return ops.non_max_suppression(
            preds,
            self.args.conf or 0.001,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
        )

    def update_metrics(self, preds, batch):
        for si, pred in enumerate(preds):
            idx = batch['batch_idx'] == si
            cls = batch['cls'][idx]
            bbox = batch['bboxes'][idx]
            nl, npr = cls.shape[0], pred.shape[0]
            shape = batch['ori_shape'][si]
            correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)
            self.seen += 1
            if npr == 0:
                if nl:
                    self.stats.append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
                continue
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            predn[:, [0, 2]] *= shape[1] / self.args.imgsz
            predn[:, [1, 3]] *= shape[0] / self.args.imgsz
            if nl:
                tbox = ops.xywh2xyxy(bbox)
                tbox[:, [0, 2]] *= shape[1]
                tbox[:, [1, 3]] *= shape[0]
                labelsn = torch.cat((cls, tbox), 1)
                correct_bboxes = self._process_batch(predn.float(), labelsn)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)
            self.stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))

    def _process_batch(self, detections, labels):
        iou = box_iou(labels[:, 1:], detections[:, :4])
        return self.match_predictions(detections[:, 5], labels[:, 0], iou)

    def get_stats(self):
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]
        if len(stats) and stats[0].any():
            self.metrics.process(*stats)
        self.nt_per_class = np.bincount(stats[3].astype(int), minlength=self.nc) if len(stats) else np.zeros(self.nc, dtype=int)
        return self.metrics.results_dict

    def finalize_metrics(self, *args, **kwargs):
        self.metrics.speed = self.speed

    def print_results(self):
        pf = '%22s' + '%11i' * 2 + '%11.3g' * 4
        LOGGER.info(pf % ('all', self.seen, self.nt_per_class.sum(), *self.metrics.box.mean_results()))
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(pf % (self.names[c], self.seen, self.nt_per_class[c], *self.metrics.box.class_result(i)))
        if self.args.plots:
            try:
                self.confusion_matrix.plot(save_dir=self.save_dir, names=list(self.names.values()), on_plot=self.on_plot)
            except Exception:
                pass

    def build_dataset(self, img_path, mode='val', batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32) if hasattr(self, 'model') else 32
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)

    def get_dataloader(self, dataset_path, batch_size):
        dataset = self.build_dataset(dataset_path, mode='val', batch=batch_size)
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)

    def plot_val_samples(self, batch, ni):
        try:
            plot_images(batch['img'], batch['batch_idx'], batch['cls'].squeeze(-1), batch['bboxes'],
                        paths=batch['im_file'], fname=self.save_dir / f'val_batch{ni}_labels.jpg',
                        names=self.names, on_plot=self.on_plot)
        except Exception:
            pass

    def plot_predictions(self, batch, preds, ni):
        try:
            plot_images(batch['img'],
                        *output_to_target(preds, max_det=self.args.max_det),
                        paths=batch['im_file'], fname=self.save_dir / f'val_batch{ni}_pred.jpg',
                        names=self.names, on_plot=self.on_plot)
        except Exception:
            pass

    def pred_to_json(self, predn, filename):
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])
        box[:, :2] -= box[:, 2:] / 2
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append({
                'image_id': image_id,
                'category_id': self.class_map[int(p[5])],
                'bbox': [round(x, 3) for x in b],
                'score': round(p[4], 5),
            })

    def save_one_txt(self, predn, save_conf, shape, file):
        from ultralytics.engine.results import Results
        Results(np.zeros((shape[0], shape[1]), dtype=np.uint8),
                path=None,
                names=self.names,
                boxes=predn[:, :6]).save_txt(file, save_conf=save_conf)


def output_to_target(output, max_det=300):
    targets, batch_idx = [], []
    for i, o in enumerate(output):
        box, conf, cls = o[:max_det, :6].cpu().split((4, 1, 1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((box, cls, conf), 1))
        batch_idx.append(j)
    return torch.cat(targets, 0), torch.cat(batch_idx, 0)
