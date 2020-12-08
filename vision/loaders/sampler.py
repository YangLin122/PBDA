import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.data import Sampler
from collections import OrderedDict

import random
import numpy as np


class TaskSampler():
	def __init__(self, unique_classes, n_way, k_shot, share=False):
		self.unique_classes = sorted(unique_classes)
		self.n_way = n_way
		self.k_shot = k_shot
		self.batch_size = n_way*k_shot
		self.counter = 0
		self.sampled_classes = None
		self.share = share

		# whether the task sampler are shared between source and target
		if self.share is True:
			self.sample_freq = 2
		else:
			self.sample_freq = 1

	def sample_N_classes(self):
		# mod-2 because both the source and target domains are using the same sampler
		# sometimes need to make sure they sample the same set of classes
		if self.counter % self.sample_freq == 0:
			self.sampled_classes = random.sample(self.unique_classes, self.n_way)

		self.counter += 1
		return self.sampled_classes

class N_Way_K_Shot_BatchSampler(Sampler):
	def __init__(self, y, max_iter, task_sampler):
		self.y = y
		self.max_iter = max_iter
		self.task_sampler = task_sampler
		self.label_dict = self.build_label_dict()
		self.unique_classes_from_y = sorted(set(self.y))

	def build_label_dict(self):
		label_dict = OrderedDict()
		for i, label in enumerate(self.y):
			if label not in label_dict:
				label_dict[label] = [i]
			else:
				label_dict[label].append(i)
		return label_dict

	def sample_examples_by_class(self, c):
		if c not in self.label_dict:
			return []

		if self.task_sampler.k_shot <= len(self.label_dict[c]):
			sampled_examples = random.sample(self.label_dict[c],
											self.task_sampler.k_shot)  # sample without replacement
		else:
			sampled_examples = random.choices(self.label_dict[c],
											k=self.task_sampler.k_shot)  # sample with replacement
		return sampled_examples

	def __iter__(self):
		for _ in range(self.max_iter):
			batch = []
			classes = self.task_sampler.sample_N_classes_as_a_task()
			for c in classes:
				samples_for_this_class = self.sample_examples_by_class(c)
				batch.extend(samples_for_this_class)

			yield batch

	def __len__(self):
		return self.max_iter


class SelfTrainingBaseSampler(Sampler):
	def __init__(self, max_iter, task_sampler, pseudo_set):
		self.max_iter = max_iter
		self.task_sampler = task_sampler
		self.pseudo_set = pseudo_set

	def __iter__(self):
		for _ in range(self.max_iter):
			batch = []
			classes = self.task_sampler.sample_N_classes_as_a_task()
			for c in classes:
				if c not in self.pseudo_set.lable_dict or not self.pseudo_set.lable_dict[c]:
					continue
				samples_for_this_class = self.sample_examples_by_class(c)
				batch.extend(samples_for_this_class)

			if len(batch) < self.task_sampler.batch_size:
				random_samples = random.sample(range(len(self.pseudo_set.set_size)), self.task_sampler.batch_size - len(batch))
				batch.extend(random_samples)

			yield batch

	def __len__(self):
		return self.max_iter


class SelfTrainingVannilaSampler(SelfTrainingBaseSampler):
	def __init__(self, max_iter, task_sampler, pseudo_set):
		super().__init__(max_iter, task_sampler, pseudo_set)

    def sample_examples_by_class(self, c):
        if self.task_sampler.k_shot <= len(self.pseudo_label_dict[c]):
            sampled_examples = random.sample(self.pseudo_label_dict[c],
                                             self.task_sampler.k_shot)  # sample without replacement
        else:
            sampled_examples = random.choices(self.pseudo_label_dict[c],
                                              k=self.task_sampler.k_shot)  # sample with replacement
        return sampled_examples


class SelfTrainingConfidentSampler(SelfTrainingBaseSampler):
	def __init__(self, max_iter, task_sampler, pseudo_set):
        super().__init__(max_iter, task_sampler, pseudo_set)

    def sample_examples_by_class(self, c):
		k = min(self.task_sampler.k_shot, len(self.pseudo_set.label_dict[c]))
		confidents = self.pseudo_set.get_wt(self.pseudo_set.lable_dict[c])
		top_k_indices_for_this_cls = confidents.argsort()[:k]
        top_k_indices = [self.pseudo_set.lable_dict[c][i] for i in top_k_indices_for_this_cls]
        return top_k_indices


class SelfTrainingUnConfidentSampler(SelfTrainingBaseSampler):
    def __init__(self, max_iter, task_sampler, pseudo_set):
        super().__init__(max_iter, task_sampler, pseudo_set)

    def sample_examples_by_class(self, cls):
		k = min(self.task_sampler.k_shot, len(self.pseudo_set.label_dict[c]))
		confidents = self.pseudo_set.get_wt(self.pseudo_set.lable_dict[c])
		top_k_indices_for_this_cls = confidents.argsort()[::-1][:k]
        top_k_indices = [self.pseudo_set.lable_dict[c][i] for i in top_k_indices_for_this_cls]
        return top_k_indices

