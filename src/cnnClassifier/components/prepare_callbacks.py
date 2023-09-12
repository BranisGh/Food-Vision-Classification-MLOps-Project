import os
import torch
import time
from torch.utils.tensorboard import SummaryWriter
from cnnClassifier.entity.config_entity import PrepareCallbacksConfig

class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig) -> None:
        self.config = config
        self.tb_root_log_dir = self.config.tensorboard_root_log_dir
        self.tb_writer = self._create_tb_callbacks()

    def _create_tb_callbacks(self) -> SummaryWriter:
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.tb_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return SummaryWriter(log_dir=tb_running_log_dir)

    def log_metrics(self, metrics) -> None:
        self.tb_writer.add_scalars('Loss', 
                                {"train": metrics.train_loss, "val": metrics.val_loss}, metrics.epoch)
        self.tb_writer.add_scalars('Accuracy', 
                                {"train": metrics.train_accuracy, "val": metrics.val_accuracy}, metrics.epoch)
        self.tb_writer.add_scalars('Precision', 
                                {"train": metrics.train_precision, "val": metrics.val_precision}, metrics.epoch)
        self.tb_writer.add_scalars('Recall', 
                                {"train": metrics.train_recall, "val": metrics.val_recall}, metrics.epoch)
        self.tb_writer.add_scalars('F1_score', 
                                {"train": metrics.train_f1_score, "val": metrics.val_f1_score}, metrics.epoch)
        self.tb_writer.add_scalars('mAP@50', 
                                {"train": metrics.train_mAP50, "val": metrics.val_mAP50}, metrics.epoch)
        self.tb_writer.add_scalars('mAP@90', 
                                {"train": metrics.train_mAP90, "val": metrics.val_mAP90}, metrics.epoch)



    def close_writer(self):
        self.tb_writer.close()
