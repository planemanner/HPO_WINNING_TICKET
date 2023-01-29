from enum import Enum
from time import time
import torch


class default_logger:
    def __init__(self, user_config):
        self.config = user_config


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, verbose_freq, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.verbose_freq = verbose_freq

    def display(self, epoch, cur_iter):
        if cur_iter % self.verbose_freq == 0:
            entries = [self.prefix + f" EPOCH : {epoch} " + self.batch_fmtstr.format(cur_iter)]
            entries += [str(meter) for meter in self.meters]
            print('\t'.join(entries))
        else:
            pass

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class classification_trainer:
    def __init__(self):
        self.global_iter = 0

    def train(self, dataloader, model, optimizer, scheduler,
              verbose_freq: int, loss_fn, scaler, tensorboard_logger,
              batch_size: int, epoch: int, rank=None):

        # Instant level loggers generation. After one-epoch, they will be re-generated.
        loss_logger = AverageMeter(name="Training Loss", fmt=":4.4f")
        time_logger = AverageMeter(name="Time", fmt=":4.4f")

        progress_meter = ProgressMeter(num_batches=len(dataloader),
                                       verbose_freq=verbose_freq,
                                       meters=[loss_logger, time_logger],
                                       prefix="Train")

        start_time = time()
        for cur_iter, data in enumerate(dataloader):
            model.train()
            input_images, labels = data

            input_images = input_images.cuda(rank)
            labels = labels.cuda(rank)

            optimizer.zero_grad()
            preds = model(input_images)
            loss = loss_fn(preds, labels)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            self.global_iter += 1

            elapsed_time = time() - start_time
            start_time = time()

            with torch.no_grad():
                if rank == 0:
                    loss_logger.update(loss.item(), n=batch_size)
                    time_logger.update(elapsed_time, n=1)
                    progress_meter.display(epoch=epoch, cur_iter=cur_iter+1)

        with torch.no_grad():
            if rank == 0:
                tensorboard_logger.writer.add_scalar(tag="Train / Loss",
                                                     scalar_value=loss_logger.avg,
                                                     global_step=epoch)

                tensorboard_logger.writer.add_scalar(tag="Training Time",
                                                     scalar_value=time_logger.avg,
                                                     global_step=epoch)
                
        scheduler.step()