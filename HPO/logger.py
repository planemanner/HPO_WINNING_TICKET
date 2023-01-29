from torch.utils.tensorboard import SummaryWriter


class tensorboard_logger:
    def __init__(self, num_classes, log_dir, logging_freq):
        self.writer = SummaryWriter(log_dir)
        self.logging_freq = logging_freq  # unit : iteration
        self.num_classes = num_classes

    def classification_logging(self, accumulated_trues, accumulated_preds, avg_top1, avg_loss, epoch):
        from sklearn.metrics import confusion_matrix
        import seaborn as sn
        from matplotlib import pyplot as plt
        import pandas as pd
        import numpy as np

        cm = confusion_matrix(accumulated_trues, accumulated_preds)
        df_cm = pd.DataFrame(cm / np.sum(cm) * self.num_classes,
                             index=[i for i in range(self.num_classes)],
                             columns=[i for i in range(self.num_classes)])

        plt.figure(figsize=(12, 7))
        figure = sn.heatmap(df_cm, annot=True).get_figure()
        self.writer.add_figure("Confusion Matrix", figure, epoch)
        self.writer.add_scalar(tag="Val / Loss", scalar_value=avg_loss, global_step=epoch)
        self.writer.add_scalar(tag="Val / Top-1", scalar_value=avg_top1, global_step=epoch)
