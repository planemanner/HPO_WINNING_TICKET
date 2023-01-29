from ._trainer import AverageMeter, ProgressMeter, Summary
import torch


def classification_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()  #

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        if len(res) == 1:
            return res[0]
        elif len(res) == 0:
            raise AssertionError("You must set proper top-k for classification_accuracy module as tuple.")
        else:
            return res


class default_evaluator:
    def __init__(self, tb_logger):
        self.eval_indices = {"values": {}}
        self.main_eval_index = None
        self.tb_logger = tb_logger

    def registry_eval_index(self):
        """
        Default:
            classification -> top1 accuracy
            object detection -> mAP (box AP)
        """
        pass

    def get_main_eval_index(self):
        pass

    def update_eval_index(self, **eval_indices):
        assert eval_indices.keys() == self.eval_indices["values"].keys(), "Check your evaluation index's names"
        for eval_index in eval_indices:
            # eval_index indicates the evaluation metric for each task. e.g., in the case of classification, top1.
            self.eval_indices["values"][eval_index] += [eval_indices[eval_index]]


class classification_evaluator(default_evaluator):
    def __init__(self, tb_logger):
        super(classification_evaluator, self).__init__(tb_logger)
        self.topk = (1,)
        self.registry_eval_index()

    def registry_eval_index(self):
        self.main_eval_index = "top1"
        self.eval_indices["values"][self.main_eval_index] = []

        """
        If you want to add some evaluation index, add that and modify related parts.
        ex)
        Step 1. add an additional evaluation index as an attribute in this method like below. 
        self.sub_eval_index = "top5"
        self.eval_indices["values"][self.sub_eval_index] = []

        Step 2.
        add 'AverageMeter' for sub_eval_index in 'validate' method .
        ex) 
        top_5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
        progress = ProgressMeter(len(dataloader),
                         verbose_freq=self.config,
                         meters=[losses, top_1, top_5],
                         prefix='Val: '
                         )
        ....
        ....
        top_5.update(acc5.item(), n=batch_size)
        self.update_eval_index(top1=top_1.avg, top5=top_5.avg) # must be kwargs format.
        And, in this case, you must modify self.topk = (1, 5).
        plus,
        acc1, acc5 = classification_accuracy(preds, labels, topk=self.topk)
        """

    def get_main_eval_index(self) -> list:
        return self.eval_indices["values"][self.main_eval_index]

    @torch.no_grad()
    def validate(self, epoch, batch_size, criterion, gpu,
                 verbose_freq, model, dataloader, ensemble=False):
        losses = AverageMeter('Loss', ':4.4f', Summary.NONE)
        top_1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(len(dataloader),
                                 verbose_freq=verbose_freq,
                                 meters=[losses, top_1],
                                 prefix='Val: '
                                 )

        model.eval()
        y_pred = []  # save predction
        y_true = []  # save ground truth
        for idx, data in enumerate(dataloader):
            images, labels = data
            images = images.cuda(gpu, non_blocking=True)
            labels = labels.cuda(gpu, non_blocking=True)
            preds = model(images)
            loss = criterion(preds, labels)

            if ensemble:
                preds = torch.mean(preds, dim=1)
            _, pred_indices = preds.topk(1, 1, True, True)
            y_pred.extend(pred_indices)  # for confusion matrix
            y_true.extend(labels)

            acc1 = classification_accuracy(preds, labels, topk=self.topk)

            top_1.update(acc1.item(), n=batch_size)
            losses.update(loss.item(), n=batch_size)

        self.update_eval_index(top1=top_1.avg)

        if gpu == 0:
            print("-Test Accuracy-")
            progress.display_summary()
            print("---------------")
            self.tb_logger.classification_logging(accumulated_trues=torch.stack(y_true).detach().cpu(),
                                                  accumulated_preds=torch.stack(y_pred).detach().cpu(),
                                                  avg_top1=top_1.avg,
                                                  avg_loss=losses.avg,
                                                  epoch=epoch)