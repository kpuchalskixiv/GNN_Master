from itertools import batched
from math import ceil
from random import random

import numpy as np
import torch
import torch.utils.data as data_utils
from tqdm import tqdm


def data_masks(all_usr_pois, item_tail):
    print("data masking start")
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)

    no_batches = 64
    batch_size = ceil(len(us_lens) / no_batches)

    all_usr_pois = batched(all_usr_pois, batch_size)
    us_lens = batched(us_lens, batch_size)
    us_msks = []
    us_pois = []

    for all_usr_pois_batch, us_lens_batch in tqdm(
        zip(all_usr_pois, us_lens), total=no_batches
    ):
        us_pois.append(
            np.asarray(
                [
                    upois + item_tail * (len_max - le)
                    for upois, le in zip(all_usr_pois_batch, us_lens_batch)
                ],
                dtype=np.uint16,
            )
        )
        us_msks.append(
            np.asarray(
                [[1] * le + [0] * (len_max - le) for le in us_lens_batch],
                dtype=np.bool_,
            )
        )

    del all_usr_pois
    del us_lens

    us_pois = np.concatenate(us_pois)
    us_msks = np.concatenate(us_msks)
    print("done masking")
    return us_pois, us_msks, len_max


class SRGNN_Dataset(data_utils.IterableDataset):
    def __init__(self, data, shuffle=False, graph=None):
        super().__init__()
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

        self.start = 0
        self.end = self.length

    def reinit(self):
        if self.end - self.start != self.length:
            self.length = self.end - self.start
            self.inputs = self.inputs[self.start : self.end]
            self.mask = self.mask[self.start : self.end]
            self.targets = self.targets[self.start : self.end]
        else:
            self.end = self.length
            self.start = 0

    def __iter__(self):
        #        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        # items, , A, alias_inputs = [], [], [], []
        assert self.start <= self.end
        assert self.end - self.start == self.length
        order = np.arange(self.length)
        if self.shuffle:
            order = np.random.permutation(order)
        n_node = []
        for u_input in self.inputs[order]:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        # print(data_utils.get_worker_info().id, self.start, self.end)
        for i, u_input in enumerate(self.inputs[order]):
            # print(i)
            node = np.unique(u_input)
            items = node.tolist()
            u_A = np.zeros((len(node), len(node)))
            for j in np.arange(len(u_input) - 1):
                if u_input[j + 1] == 0:
                    break
                u = np.where(node == u_input[j])[0][0]
                v = np.where(node == u_input[j + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A = np.pad(
                u_A,
                (
                    (0, max_n_node - node.shape[0]),
                    (0, 2 * (max_n_node - node.shape[0])),
                ),
            )
            alias_inputs = np.asarray([np.where(node == i)[0][0] for i in u_input])

            items = np.pad(items, (0, max_n_node - node.shape[0]))

            yield alias_inputs, A, items, self.mask[i], self.targets[i]


class SRGNN_Map_Dataset(data_utils.Dataset):
    def __init__(
        self,
        data,
        shuffle=False,
        graph=None,
        noise_p=0.0,
        noise_mean=0.01,
        noise_std=0.0,
    ):
        super().__init__()
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        # self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

        self.start = 0
        self.end = self.length
        self.noise_p = noise_p
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def reinit(self):
        if self.end - self.start != self.length:
            self.length = self.end - self.start
            self.inputs = self.inputs[self.start : self.end]
            self.mask = self.mask[self.start : self.end]
            self.targets = self.targets[self.start : self.end]
        else:
            self.end = self.length
            self.start = 0

    def __len__(self):
        return self.length

    def blur_and_pad_matrix(self, M, max_n_node):
        assert (
            M.shape[0] == M.shape[1]
        )  # this is in/out part of adjacency matrix, should be square
        n = M.shape[0]

        if self.noise_std:
            noise_M = np.random.random(size=(n, n)) < self.noise_p
            M += noise_M * np.random.normal(
                loc=self.noise_mean, scale=self.noise_std, size=(n, n)
            )
        M = np.pad(M, pad_width=(0, max_n_node - n), constant_values=0)
        return M

    def _collect_correct_samples(self, idxs):
        if isinstance(idxs, int):
            idxs = [idxs]
        # print(idxs)
        inputs, mask, targets = self.inputs[idxs], self.mask[idxs], self.targets[idxs]
        #   non_zero_cols = (mask != 0).sum(axis=0) != 0
        #   inputs = inputs[:, non_zero_cols]
        #   mask = mask[:, non_zero_cols]
        max_zero_idx = np.argmin((mask != 0).sum(axis=0))
        inputs = np.pad(inputs[:, :max_zero_idx], ((0, 0), (0, 1)), constant_values=0)
        mask = np.pad(mask[:, :max_zero_idx], ((0, 0), (0, 1)), constant_values=0)

        n_node = []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)  # length of the longest session in batch + padding

        return inputs, mask, targets, max_n_node

    def __getitem__(self, idxs):

        inputs, mask, targets, max_n_node = self._collect_correct_samples(idxs)
        items, A, alias_inputs = [], [], []

        for u_input in inputs:
            node = np.unique(u_input)
            items.append(np.concatenate([node, np.zeros(max_n_node - len(node))]))
            u_A = np.zeros((len(node), len(node)))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1

            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)

            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)

            u_A_in = self.blur_and_pad_matrix(u_A_in, max_n_node)
            u_A_out = self.blur_and_pad_matrix(u_A_out, max_n_node)

            u_A = np.concatenate([u_A_in, u_A_out]).transpose()

            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return (
            np.asarray(alias_inputs),
            np.asarray(A),
            np.asarray(items),
            np.asarray(mask),
            targets,
        )


def worker_init_fn(worker_id):
    worker_info = data_utils.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = 0
    overall_end = dataset.length
    # configure the dataset to only process the split workload
    per_worker = int(
        ceil((overall_end - overall_start) / float(worker_info.num_workers))
    )
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)
    dataset.reinit()


class SRGNN_sampler(data_utils.Sampler):
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self):
        return self.dataset.length

    def __iter__(self):
        order = np.arange(len(self))
        if self.shuffle:
            np.random.shuffle(order)
        if len(self) % self.batch_size:
            for i in range(0, len(self) - self.batch_size, self.batch_size):
                yield order[i : i + self.batch_size]
            if not self.drop_last:
                yield order[-(len(self) % self.batch_size) :]
        else:
            for i in range(0, len(self), self.batch_size):
                yield order[i : i + self.batch_size]

    # raise IndexError('Done iterating')


class AugmentDataset(SRGNN_Map_Dataset):
    def __init__(
        self,
        item_labels=None,
        cluster_centers=None,
        clip=0,
        normalize=False,
        p=1.0,
        noise_p=1.0,
        noise_mean=0.01,
        noise_std=0.0,
        prenormalize_distances=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.item_labels = item_labels
        self.cluster_centers = cluster_centers

        self.clip = clip
        self.normalize = normalize
        self.prenormalize_distances = prenormalize_distances

        self.p = p
        self.noise_p = noise_p
        self.noise_mean = noise_mean
        self.noise_std = noise_std

        assert (not normalize) or (not clip), "Usage of both not implemented!"

    def postprocess_matrix(self, u_A, loops, max_n_node):
        if self.normalize:
            maxes = 2 * np.max(u_A, 0)
            u_A_in = u_A.copy()
            for u, v in loops:
                u_A_in[u, v] = max(1, maxes[u])

            u_sum_in = np.sum(u_A_in, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A_in, u_sum_in)

            maxes = 2 * np.max(u_A, 1)
            for u, v in loops:
                u_A[u, v] = max(1, maxes[u])

            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
        elif self.clip:
            for u, v in loops:
                u_A[u, v] = self.clip
            u_A = np.clip(u_A, a_min=0, a_max=self.clip)
            u_A_sum = np.sum(u_A)
            if u_A_sum:
                u_A_in = u_A / u_A_sum
                u_A_out = u_A.transpose() / u_A_sum
            else:
                u_A_in = u_A.copy()
                u_A_out = u_A.transpose()
        else:
            maxes = 2 * np.max(u_A, 0)
            for u, v in loops:
                u_A[u, v] = max(1, maxes[u])

            u_A_in = u_A.copy()
            u_A_out = u_A.transpose()

        u_A_in = self.blur_and_pad_matrix(u_A_in, max_n_node)
        u_A_out = self.blur_and_pad_matrix(u_A_out, max_n_node)
        u_A = np.concatenate([u_A_in, u_A_out]).transpose()
        return u_A


class Augment_Matrix_Dataset(
    AugmentDataset
):  # TODO change name to sth like item2item augment dataset
    def __init__(
        self,
        emb_model=None,
        clip=0,
        normalize=False,
        p=1.0,
        noise_mean=0.01,
        noise_std=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.emb_model = emb_model
        self.emb_model.to("cpu")
        self.emb_model.eval()
        self.clip = clip
        self.normalize = normalize
        self.p = p
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        assert (not normalize) or (not clip), "Usage of both not implemented!"

    def __getitem__(self, idxs):
        inputs, mask, targets, max_n_node = self._collect_correct_samples(idxs)
        items, A, alias_inputs = [], [], []

        for u_input in inputs:
            loops = []
            node = np.unique(u_input)
            item_embeddigs = (
                self.emb_model(
                    torch.tensor(np.asarray(node, dtype=np.int32), device="cpu")
                )
                .cpu()
                .detach()
                .numpy()
            )
            items.append(np.concatenate([node, np.zeros(max_n_node - len(node))]))
            u_A = np.zeros((len(node), len(node)))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                if random() < self.p:
                    if u == v:
                        loops.append((u, v))  # for compatibility
                        continue
                    u_A[u][v] = 1 / np.linalg.norm(
                        item_embeddigs[u] - item_embeddigs[v]
                    )
                else:
                    u_A[u][v] = 1

            u_A = self.postprocess_matrix(u_A, loops, max_n_node)
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return (
            np.asarray(alias_inputs),
            np.asarray(A),
            np.asarray(items),
            np.asarray(mask),
            targets,
        )


class Clusters_Matrix_Dataset(AugmentDataset):
    def __init__(
        self,
        item_labels=None,
        cluster_centers=None,
        clip=0,
        normalize=False,
        p=1.0,
        noise_p=1.0,
        noise_mean=0.01,
        noise_std=0.0,
        prenormalize_distances=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.item_labels = item_labels
        self.cluster_centers = cluster_centers

        self.clip = clip
        self.normalize = normalize
        self.prenormalize_distances = prenormalize_distances

        self.p = p
        self.noise_p = noise_p
        self.noise_mean = noise_mean
        self.noise_std = noise_std

        assert (not normalize) or (not clip), "Usage of both not implemented!"

        no_clusters = len(cluster_centers)
        self.cluster_distances = np.zeros((no_clusters, no_clusters))

        if not isinstance(cluster_centers, dict):
            for i in range(no_clusters):
                self.cluster_distances[i] = np.linalg.norm(
                    cluster_centers - cluster_centers[i], axis=1
                )
            self.cluster_distances = 1 / self.cluster_distances
            self.cluster_distances[self.cluster_distances == np.inf] = 0
            s = self.cluster_distances.sum(axis=1)
            self.cluster_distances = self.cluster_distances / s.reshape(no_clusters, 1)
            self.cluster_distances = self.cluster_distances / self.cluster_distances.max()

    def __getitem__(self, idxs):
        inputs, mask, targets, max_n_node = self._collect_correct_samples(idxs)
        items, A, alias_inputs = [], [], []

        for u_input in inputs:
            loops = []
            node = np.unique(u_input)
            items.append(np.concatenate([node, np.zeros(max_n_node - len(node))]))
            u_A = np.zeros((len(node), len(node)))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]

                if random() < self.p:
                    u_label = self.item_labels[u_input[i]]
                    v_label = self.item_labels[u_input[i + 1]]
                    if u_label == v_label:
                        loops.append((u, v))
                        continue
                    if self.prenormalize_distances:
                        u_A[u][v] = self.cluster_distances[u, v]
                    else:
                        u_A[u][v] = 1 / np.linalg.norm(
                            self.cluster_centers[u_label]
                            - self.cluster_centers[v_label]
                        )
                else:
                    u_A[u][v] = 1

            u_A = self.postprocess_matrix(u_A, loops, max_n_node)
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return (
            np.asarray(alias_inputs),
            np.asarray(A),
            np.asarray(items),
            np.asarray(mask),
            targets,
        )
