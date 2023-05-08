from typing import List

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import pairwise_euclidean_distance

from . import metric_utils


# motion reconstruction metric
class ReconstructionMetrics(Metric):
    def __init__(
        self,
        njoints,
        jointstype: str = "mmm",
        force_in_meter: bool = True,
        align_root: bool = True,
        dist_sync_on_step=True,
        **kwargs
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if jointstype not in ["mmm", "humanml3d"]:
            raise NotImplementedError("This jointstype is not implemented.")

        self.name = "Motion Reconstructions"
        self.jointstype = jointstype
        self.align_root = align_root
        self.force_in_meter = force_in_meter

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state("MPJPE", default=torch.tensor([0.0]), dist_reduce_fx="sum")
        self.add_state("PAMPJPE", default=torch.tensor([0.0]), dist_reduce_fx="sum")
        self.add_state("ACCEL", default=torch.tensor([0.0]), dist_reduce_fx="sum")
        # todo
        # self.add_state("ROOT", default=torch.tensor([0.0]), dist_reduce_fx="sum")

        self.MR_metrics = ["MPJPE", "PAMPJPE", "ACCEL"]

        # All metric
        self.metrics = self.MR_metrics

    def compute(self):
        if self.force_in_meter:
            # different jointstypes have different scale factors
            # if self.jointstype == 'mmm':
            #     factor = 1000.0
            # elif self.jointstype == 'humanml3d':
            #     factor = 1000.0 * 0.75 / 480
            factor = 1000.0
        else:
            factor = 1.0

        count = self.count
        count_seq = self.count_seq
        mr_metrics = {}
        mr_metrics["MPJPE"] = self.MPJPE / count * factor
        mr_metrics["PAMPJPE"] = self.PAMPJPE / count * factor
        # accel error: joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
        # n-2 for each sequences
        mr_metrics["ACCEL"] = self.ACCEL / (count - 2 * count_seq) * factor
        return mr_metrics

    def update(self, joints_rst: Tensor, joints_ref: Tensor, lengths: List[int]):
        assert joints_rst.shape == joints_ref.shape
        assert joints_rst.dim() == 4
        # (bs, seq, njoint=22, 3)

        self.count += sum(lengths)
        self.count_seq += len(lengths)

        # avoid cuda error of DDP in pampjpe
        rst = joints_rst.detach().cpu()
        ref = joints_ref.detach().cpu()

        # align root joints index
        if self.align_root and self.jointstype in ["mmm", "humanml3d"]:
            align_inds = [0]
        else:
            align_inds = None

        for i in range(len(lengths)):
            self.MPJPE += torch.sum(
                metric_utils.calc_mpjpe(rst[i], ref[i], align_inds=align_inds)
            )
            self.PAMPJPE += torch.sum(metric_utils.calc_pampjpe(rst[i], ref[i]))
            self.ACCEL += torch.sum(metric_utils.calc_accel(rst[i], ref[i]))


class UnconditonalGenerationMetrics(Metric):
    full_state_update = True

    def __init__(
        self, top_k=3, R_size=32, diversity_times=300, dist_sync_on_step=True, **kwargs
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "fid, kid, and diversity scores"

        self.top_k = top_k
        self.R_size = R_size
        self.diversity_times = 300

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")

        self.metrics = []

        # KID
        self.add_state("KID_mean", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("KID_std", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.metrics.extend(["KID_mean", "KID_std"])
        # Fid
        self.add_state("FID", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.metrics.append("FID")

        # Diversity
        self.add_state("Diversity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("gt_Diversity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.extend(["Diversity", "gt_Diversity"])

        # chached batches
        self.add_state("recmotion_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("gtmotion_embeddings", default=[], dist_reduce_fx=None)

    def compute(self):
        count = self.count.item()
        count_seq = self.count_seq.item()

        # init metrics
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # cat all embeddings
        all_gtmotions = torch.cat(self.gtmotion_embeddings, axis=0).cpu()
        all_genmotions = torch.cat(self.recmotion_embeddings, axis=0).cpu()

        # Compute kid

        KID_mean, KID_std = metric_utils.calculate_kid(all_gtmotions, all_genmotions)
        metrics["KID_mean"] = KID_mean
        metrics["KID_std"] = KID_std

        # tensor -> numpy for FID
        all_genmotions = all_genmotions.numpy()
        all_gtmotions = all_gtmotions.numpy()

        # Compute fid
        mu, cov = metric_utils.calculate_activation_statistics_np(all_genmotions)
        # gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)
        gt_mu, gt_cov = metric_utils.calculate_activation_statistics_np(all_gtmotions)
        metrics["FID"] = metric_utils.calculate_frechet_distance_np(
            gt_mu, gt_cov, mu, cov
        )

        # Compute diversity
        assert count_seq > self.diversity_times
        metrics["Diversity"] = metric_utils.calculate_diversity_np(
            all_genmotions, self.diversity_times
        )
        metrics["gt_Diversity"] = metric_utils.calculate_diversity_np(
            all_gtmotions, self.diversity_times
        )

        return metrics

    def update(
        self,
        gtmotion_embeddings: Tensor,
        recmotion_embeddings: Tensor,
        lengths: List[int],
    ):
        self.count += sum(lengths)
        self.count_seq += len(lengths)

        # [bs, nlatent*ndim] <= [bs, nlatent, ndim]
        recmotion_embeddings = torch.flatten(recmotion_embeddings, start_dim=1).detach()
        # store all texts and motions
        self.recmotion_embeddings.append(recmotion_embeddings)
        gtmotion_embeddings = torch.flatten(gtmotion_embeddings, start_dim=1).detach()

        self.gtmotion_embeddings.append(gtmotion_embeddings)
