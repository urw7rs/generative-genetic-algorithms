from jax.typing import ArrayLike

import jax
import jax.numpy as jnp
from jax.scipy import linalg

from flax import struct


# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
    Params:
    -- matrix1: N1 x D
    -- matrix2: N2 x D
    Returns:
    -- dist: N1 x N2
    dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * jnp.dot(matrix1, matrix2.T)  # shape (num_test, num_train)
    d2 = jnp.sum(jnp.square(matrix1), axis=1, keepdims=True)  # shape (num_test, 1)
    d3 = jnp.sum(jnp.square(matrix2), axis=1)  # shape (num_train, )
    dists = jnp.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = jnp.expand_dims(jnp.arange(size), 1).repeat(size, 1)
    bool_mat = mat == gt_mat
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        #         print(correct_vec, bool_mat[:, i])
        correct_vec = correct_vec | bool_mat[:, i]
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = jnp.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    argmax = jnp.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    else:
        return top_k_mat


def calculate_matching_score(embedding1, embedding2, sum_all=False):
    assert len(embedding1.shape) == 2
    assert embedding1.shape[0] == embedding2.shape[0]
    assert embedding1.shape[1] == embedding2.shape[1]

    dist = linalg.norm(embedding1 - embedding2, axis=1)
    if sum_all:
        return dist.sum(axis=0)
    else:
        return dist


def calculate_activation_statistics(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = jnp.mean(activations, axis=0)
    cov = jnp.cov(activations, rowvar=False)
    return mu, cov


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = jax.random.choice(num_samples, diversity_times, replace=False)
    second_indices = jax.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_multimodality(activation, multimodality_times, key):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    key, first_key, second_key = jax.random.split(key, 3)
    first_dices = jax.random.choice(
        first_key, num_per_sent, multimodality_times, replace=False
    )
    second_dices = jax.random.choice(
        second_key, num_per_sent, multimodality_times, replace=False
    )
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative dataset set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative dataset set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = jnp.atleast_1d(mu1)
    mu2 = jnp.atleast_1d(mu2)

    sigma1 = jnp.atleast_2d(sigma1)
    sigma2 = jnp.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not jnp.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = jnp.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if jnp.iscomplexobj(covmean):
        if not jnp.allclose(jnp.diagonal(covmean).imag, 0, atol=1e-3):
            m = jnp.max(jnp.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = jnp.trace(covmean)

    return diff.dot(diff) + jnp.trace(sigma1) + jnp.trace(sigma2) - 2 * tr_covmean


def compute_mpjpe(preds, target, valid_mask=None, pck_joints=None, sample_wise=True):
    """
    Mean per-joint position error (i.e. mean Euclidean distance)
    often referred to as "Protocol #1" in many papers.
    """
    assert preds.shape == target.shape, print(preds.shape, target.shape)  # BxJx3
    mpjpe = jnp.linalg.norm(preds - target, ord=2, axis=-1)  # BxJ

    if pck_joints is None:
        if sample_wise:
            mpjpe_seq = (
                (mpjpe * valid_mask).sum(-1) / valid_mask.sum(-1)
                if valid_mask is not None
                else mpjpe.mean(-1)
            )
        else:
            mpjpe_seq = mpjpe[valid_mask] if valid_mask is not None else mpjpe
        return mpjpe_seq
    else:
        mpjpe_pck_seq = mpjpe[:, pck_joints]
        return mpjpe_pck_seq


def align_by_parts(joints, align_indices=None):
    if align_indices is None:
        return joints
    pelvis = joints[:, align_indices].mean(1)
    return joints - pelvis[:, None]


def calc_mpjpe(preds, target, align_indices=[0], sample_wise=True, trans=None):
    # Expects BxJx3
    valid_mask = target[:, :, 0] != -2.0
    # valid_mask = torch.BoolTensor(target[:, :, 0].shape)
    if align_indices is not None:
        preds_aligned = align_by_parts(preds, align_indices=align_indices)
        if trans is not None:
            preds_aligned += trans
        target_aligned = align_by_parts(target, align_indices=align_indices)
    else:
        preds_aligned, target_aligned = preds, target
    mpjpe_each = compute_mpjpe(
        preds_aligned, target_aligned, valid_mask=valid_mask, sample_wise=sample_wise
    )
    return mpjpe_each


@jax.vmap
def batched_mpjpe(preds, target):
    return calc_mpjpe(preds, target)


@jax.vmap
def batched_accel(preds, target):
    """
    Mean joint acceleration error
    often referred to as "Protocol #1" in many papers.
    """
    assert preds.shape == target.shape  # BxJx3
    assert preds.dim() == 3
    # Expects BxJx3
    # valid_mask = torch.BoolTensor(target[:, :, 0].shape)
    accel_gt = target[:-2] - 2 * target[1:-1] + target[2:]
    accel_pred = preds[:-2] - 2 * preds[1:-1] + preds[2:]
    normed = jnp.linalg.norm(accel_pred - accel_gt, axis=-1)
    accel_seq = normed.mean(1)
    return accel_seq


def bmm(a, b):
    return jax.lax.batch_matmul(a, b, precision="highest")


@jax.vmap
def svd(X):
    return linalg.svd(X, full_matrices=False, lapack_driver="gesvd")


def similarity_transform(S1: ArrayLike, S2: ArrayLike):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    S1 = S1.transpose((0, 2, 1))
    S2 = S2.transpose((0, 2, 1))

    assert S2.shape[1] == S1.shape[1]

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = jnp.sum(X1**2, axis=1).sum(axis=1)

    # 3. The outer product of X1 and X2.
    K = bmm(X1, X2.transpose((0, 2, 1)))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = jnp.linalg.svd(K, full_matrices=False)
    # U, s, V = svd(K)
    V = V.transpose((0, 2, 1))

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = jnp.eye(U.shape[1])[None, :]
    Z = jnp.tile(Z, (U.shape[0], 1, 1))
    Z = Z.at[:, -1, -1].set(
        Z[:, -1, -1]
        * jnp.sign(
            jnp.linalg.det(
                bmm(U, V.transpose((0, 2, 1))),
            )
        )
    )

    # Construct R.
    R = bmm(V, bmm(Z, U.transpose((0, 2, 1))))

    # 5. Recover scale.
    scale = jnp.trace(bmm(R, K), axis1=1, axis2=2) / var1

    # 6. Recover translation.
    t = mu2 - (scale[..., None, None] * bmm(R, mu1))

    # 7. Error:
    S1_hat = scale[..., None, None] * bmm(R, S1) + t

    S1_hat = S1_hat.transpose((0, 2, 1))

    return S1_hat, (scale, R, t)


@jax.vmap
def batched_pampjpe(preds, target):
    # Expects BxJx3
    # extracting the keypoints that all samples have valid annotations
    # valid_mask = (target[:, :, 0] != -2.).sum(0) == len(target)
    # preds_tranformed, PA_transform = batch_compute_similarity_transform_torch(preds[:, valid_mask], target[:, valid_mask])
    # pa_mpjpe_each = compute_mpjpe(preds_tranformed, target[:, valid_mask], sample_wise=sample_wise)

    preds_tranformed, PA_transform = similarity_transform(preds, target)
    pa_mpjpe_each = compute_mpjpe(preds_tranformed, target, sample_wise=True)

    return pa_mpjpe_each


@struct.dataclass
class ReconstructionMetrics:
    mpjpe: jax.Array
    pampjpe: jax.Array
    accel: jax.Array
    count: int
    seq_count: int
    align_root: bool

    def update(self, targets, predictions):
        self.mpjpe += batched_mpjpe(predictions, targets).sum()
        self.pampjpe += batched_pampjpe(predictions, targets).sum()
        self.accel += batched_accel(predictions, targets).sum()

    def compute(self):
        return self.mpjpe


@struct.dataclass
class GenerationMetrics:
    targets: jax.Array
    predictions: jax.Array

    def update(self, targets, predictions):
        ...


def update(state, targets, predictions):
    targets = jnp.concatenate([state.targets, targets], axis=0)
    preds = jnp.concatenate([state.predictions, predictions], axis=0)
    return GenerationMetrics(targets, preds)


def compute_fid(state):
    mu, cov = calculate_activation_statistics(state.preds)
    gt_mu, gt_cov = calculate_activation_statistics(state.targets)

    fid_score = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    return fid_score


def compute_diversity(state, n=300):
    diversity = calculate_diversity(state.preds, n)
    gt_diversity = calculate_diversity(state.targets, n)
    return diversity, gt_diversity
