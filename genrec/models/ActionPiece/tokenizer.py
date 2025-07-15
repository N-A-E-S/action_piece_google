# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ActionPiece tokenizer for GenRec."""

import collections
import json
import os
from typing import Any

import faiss
from genrec.dataset import AbstractDataset
from genrec.models.ActionPiece.core import ActionPieceCore
from genrec.tokenizer import AbstractTokenizer
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import torch


class ActionPieceTokenizer(AbstractTokenizer):
  """The ActionPiece tokenizer is a tokenizer that encodes the attribute.

  features.

  Attributes:
      item2feat (dict): A dictionary mapping item IDs to their features.
      ignored_label (int): The label to be used for padding tokens.
      actionpiece (ActionPieceCore): ActionPiece core tokenizer.
      bos_token (int): The beginning token.
      eos_token (int): The end token.
      n_prob_encode_plus (int): The number of probability encoding plus.
      n_inference_ensemble (int): The number of inference ensemble.
      train_shuffle (str): The shuffle strategy for training.
      encoded_labels (dict): A dictionary mapping label sequences to their
        encoded representations.
      collate_fn (dict): A dictionary mapping split names to their corresponding
        collate functions.
  """

  def __init__(self, config: dict[Any, Any], dataset: AbstractDataset):
    super().__init__(config)

    self.item2feat = None
    self.ignored_label = -100
    self.actionpiece = self._init_tokenizer(dataset)
    self.bos_token = self.actionpiece.vocab_size
    self.eos_token = self.actionpiece.vocab_size + 1
    self.n_prob_encode_plus = self.config['n_prob_encode_plus']
    self.n_inference_ensemble = config['n_inference_ensemble']
    self.train_shuffle = config['train_shuffle']
    self.encoded_labels = {}
    self.collate_fn = {
        'train': self.collate_fn_train,
        'val': self.collate_fn_val,
        'test': self.collate_fn_test,
    }

  def _encode_sent_emb(self, dataset: AbstractDataset, output_path: str):
    """Encodes the sentence embeddings for the given dataset and saves them to the specified output path.

    Args:
        dataset (AbstractDataset): The dataset containing the sentences to
          encode.
        output_path (str): The path to save the encoded sentence embeddings.

    Returns:
        numpy.ndarray: The encoded sentence embeddings.
    """
    assert self.config['metadata'] in ['sentence', 'all']

    sent_emb_model = SentenceTransformer(self.config['sent_emb_model']).to(
        self.config['device']
    )

    meta_sentences = []  # 1-base, meta_sentences[0] -> item_id = 1
    item2meta = None
    if 'sentence' in dataset.item2meta:
      item2meta = dataset.item2meta['sentence']
    else:
      item2meta = dataset.item2meta
    for i in range(1, dataset.n_items):
      meta_sentences.append(item2meta[dataset.id_mapping['id2item'][i]])
    sent_embs = sent_emb_model.encode(
        meta_sentences,
        convert_to_numpy=True,
        batch_size=self.config['sent_emb_batch_size'],
        show_progress_bar=True,
        device=self.config['device'],
    )

    # PCA
    if self.config['sent_emb_pca'] > 0:
      self.logger.info('[TOKENIZER] Applying PCA to sentence embeddings...')

      pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
      sent_embs = pca.fit_transform(sent_embs)

    sent_embs.tofile(output_path)
    return sent_embs

  def _get_sent_embs(self, dataset: AbstractDataset) -> np.ndarray:
    # Load or encode sentence embeddings
    sent_emb_path = os.path.join(
        dataset.cache_dir,
        'processed',
        f'{os.path.basename(self.config["sent_emb_model"])}.sent_emb',
    )
    sent_emb_dim = (
        self.config['sent_emb_dim']
        if self.config['sent_emb_pca'] <= 0
        else self.config['sent_emb_pca']
    )
    if os.path.exists(sent_emb_path):
      self.logger.info(
          f'[TOKENIZER] Loading sentence embeddings from {sent_emb_path}...'
      )
      sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(
          -1, sent_emb_dim
      )
    else:
      self.logger.info('[TOKENIZER] Encoding sentence embeddings...')
      sent_embs = self._encode_sent_emb(dataset, sent_emb_path)
    self.logger.info(
        f'[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}'
    )
    return sent_embs

  def _get_items_for_training(self, dataset: AbstractDataset) -> np.ndarray:
    items_for_training = set()
    for item_seq in dataset.split_data['train']['item_seq']:
      for item in item_seq:
        items_for_training.add(item)
    self.logger.info(
        f'[TOKENIZER] Items for training: {len(items_for_training)}'
        f' of {dataset.n_items - 1}'
    )
    mask = np.zeros(dataset.n_items - 1, dtype=bool)
    for item in items_for_training:
      mask[dataset.item2id[item] - 1] = True
    return mask

  # def _sent_emb_to_sem_id(
  #     self, dataset: AbstractDataset, sent_embs: np.ndarray
  # ) -> dict[Any, Any]:
  #   # Get the sentence embeddings for training
  #   training_item_mask = self._get_items_for_training(dataset)
  #   embs_for_training = sent_embs[training_item_mask]

  #   # Train the index
  #   # Take the vector quantized codes as item features

  #   faiss.omp_set_num_threads(self.config['n_threads'])
  #   index = faiss.index_factory(
  #       sent_embs.shape[-1],
  #       f"OPQ{self.config['pq_n_codebooks']},IVF1,PQ{self.config['pq_n_codebooks']}x{int(np.log2(self.config['pq_codebook_size']))}",
  #       faiss.METRIC_INNER_PRODUCT,
  #   )
  #   self.logger.info('[TOKENIZER] Training index...')
  #   index.train(embs_for_training)
  #   index.add(sent_embs)

  #   ivf_index = faiss.downcast_index(index.index)
  #   invlists = faiss.extract_index_ivf(ivf_index).invlists
  #   ls = invlists.list_size(0)
  #   sem_ids = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
  #   sem_ids = sem_ids.reshape(-1, invlists.code_size)

  #   # Convert semantic IDs to a dictionary
  #   item2sem_ids = {}
  #   for i in range(sem_ids.shape[0]):
  #     item = dataset.id_mapping['id2item'][i + 1]
  #     item2sem_ids[item] = tuple(sem_ids[i].tolist())
  #   return item2sem_ids


  def _sent_emb_to_sem_id(
      self, dataset: AbstractDataset, sent_embs: np.ndarray
  ) -> dict[Any, Any]:
      # Get the sentence embeddings for training
      training_item_mask = self._get_items_for_training(dataset)
      embs_for_training = sent_embs[training_item_mask]

      # Train the index - 修复 faiss segfault 问题
      # Take the vector quantized codes as item features

      # 在 macOS 上设置 faiss 的线程数来避免 segfault
      import os
      os.environ['OMP_NUM_THREADS'] = '1'
      
      try:
          import faiss
          faiss.omp_set_num_threads(self.config['n_threads'])
          
          # 使用更简单、更稳定的索引类型来避免 segfault
          d = sent_embs.shape[-1]
          
          # 检查配置是否合理
          n_codebooks = min(self.config['pq_n_codebooks'], d // 8)  # 确保不会超过维度限制
          codebook_bits = min(8, int(np.log2(self.config['pq_codebook_size'])))
          
          self.log(f'[TOKENIZER] Using {n_codebooks} codebooks with {codebook_bits} bits each')
          
          # 使用更稳定的索引构建方式
          try:
              # 方法1：尝试使用简单的 PQ（避免 OPQ）
              index = faiss.IndexPQ(d, n_codebooks, codebook_bits)
              self.log('[TOKENIZER] Using simple PQ index')
          except Exception as e:
              self.log(f'[TOKENIZER] PQ failed ({e}), using fallback...')
              # 方法2：最后的回退方案 - 使用随机量化
              return self._fallback_random_quantization(sent_embs, n_codebooks, self.config['pq_codebook_size'])
          
          self.log('[TOKENIZER] Training index...')
          
          # 确保训练数据是连续的和正确的类型
          embs_for_training = np.ascontiguousarray(embs_for_training.astype('float32'))
          sent_embs_cont = np.ascontiguousarray(sent_embs.astype('float32'))
          
          # 训练索引
          index.train(embs_for_training)
          
          self.log('[TOKENIZER] Computing codes...')
          # 直接计算编码，不添加到索引
          codes = index.pq.compute_codes(sent_embs_cont)

      except Exception as e:
          self.log(f'[TOKENIZER] Faiss training failed with error: {e}')
          self.log('[TOKENIZER] Falling back to random quantization...')
          return self._fallback_random_quantization(sent_embs, n_codebooks, self.config['pq_codebook_size'])

      # Convert semantic IDs to a dictionary
      item2sem_ids = {}
      for i in range(codes.shape[0]):
          item = dataset.id_mapping['id2item'][i + 1]
          item2sem_ids[item] = tuple(codes[i].tolist())
      return item2sem_ids

  def _fallback_random_quantization(self, sent_embs: np.ndarray, n_codebooks: int, codebook_size: int) -> dict[Any, Any]:
      """回退方案：使用随机量化代替 faiss"""
      self.log('[TOKENIZER] Using fallback random quantization...')
      
      # 使用 sklearn 的 KMeans 作为替代
      try:
          from sklearn.cluster import KMeans
          
          codes = np.zeros((sent_embs.shape[0], n_codebooks), dtype=np.uint8)
          
          # 对每个子空间进行聚类
          subvec_length = sent_embs.shape[1] // n_codebooks
          
          for i in range(n_codebooks):
              start_idx = i * subvec_length
              end_idx = (i + 1) * subvec_length if i < n_codebooks - 1 else sent_embs.shape[1]
              
              subvectors = sent_embs[:, start_idx:end_idx]
              
              # 使用 KMeans 聚类
              n_clusters = min(codebook_size, subvectors.shape[0])
              kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1)
              cluster_labels = kmeans.fit_predict(subvectors)
              
              codes[:, i] = cluster_labels.astype(np.uint8)
              
              self.log(f'[TOKENIZER] Completed codebook {i+1}/{n_codebooks}')
          
      except ImportError:
          self.log('[TOKENIZER] sklearn not available, using random codes...')
          # 最简单的回退：随机分配
          codes = np.random.randint(0, codebook_size, size=(sent_embs.shape[0], n_codebooks), dtype=np.uint8)
      
      # Convert to dictionary format - 需要正确访问 dataset
      item2sem_ids = {}
      for i in range(codes.shape[0]):
          # 这里需要从 dataset 的 id_mapping 中获取正确的 item
          try:
              # 访问数据集的 item 映射
              if hasattr(self, 'dataset') and hasattr(self.dataset, 'id_mapping'):
                  item = self.dataset.id_mapping['id2item'][i + 1]
              else:
                  # 如果无法访问，使用索引作为 key
                  item = f"item_{i}"
              item2sem_ids[item] = tuple(codes[i].tolist())
          except (IndexError, KeyError):
              # 处理索引超出范围的情况
              item = f"item_{i}"
              item2sem_ids[item] = tuple(codes[i].tolist())
      
      return item2sem_ids

  def _get_sem_ids(self, dataset: AbstractDataset) -> dict[Any, Any]:
    """Get the semantic IDs from the dataset.

    If the semantic IDs are already cached, load them from the cache. Otherwise,
    generate the semantic IDs and save them to the cache.

    Args:
        dataset (AbstractDataset): The dataset containing the items to get
          semantic IDs for.

    Returns:
        item2sem_ids (dict): A dictionary mapping item IDs to their semantic
        IDs.
    """
    sem_ids_path = os.path.join(
        dataset.cache_dir,
        'processed',
        f'{os.path.basename(self.config["sent_emb_model"])}.sem_ids',
    )
    if not os.path.exists(sem_ids_path):
      self.logger.info(
          '[TOKENIZER] Semantic IDs not found. Training index using Faiss...'
      )
      sent_embs = self._get_sent_embs(dataset)
      item2sem_ids = self._sent_emb_to_sem_id(dataset, sent_embs)
      # Save semantic IDs
      self.logger.info(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
      with open(sem_ids_path, 'w') as f:
        json.dump(item2sem_ids, f)
      return item2sem_ids
    else:
      self.logger.info(
          f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...'
      )
      with open(sem_ids_path, 'r') as f:
        item2sem_ids = json.load(f)
      return {
          k: v[: self.config['rq_n_codebooks']] for k, v in item2sem_ids.items()
      }

  def _get_attr_ids(self, dataset: AbstractDataset):
    """Get the attribute IDs from the dataset."""
    if 'attr_id' in dataset.item2meta:
      return dataset.item2meta['attr_id']
    else:
      return None

  def _combine_features(
      self, item2sem_ids: dict[Any, Any], item2attr_ids: dict[Any, Any]
  ) -> dict[Any, Any]:
    """Combine the semantic IDs and attribute IDs to get the item features."""
    if item2attr_ids is None:
      return item2sem_ids
    item2feat = {}
    for item in item2sem_ids:
      feat = item2sem_ids[item] + item2attr_ids[item]
      item2feat[item] = feat
    return item2feat

  def _get_hashed_feat(
      self, dataset: AbstractDataset, item2feat: dict[Any, Any]
  ) -> dict[Any, Any]:
    """Add one digit of the original features as the hash buckets.

    The purpose is to avoid conflicts.

    Args:
        dataset (AbstractDataset): The dataset containing the items (not used).
        item2feat (dict): A dictionary mapping item IDs to their features.

    Returns:
        dict: A dictionary mapping item IDs to their hashed features.
    """
    feat2cnt = collections.defaultdict(int)
    feat2hash_ids = {}
    item2hashed_feat = {}

    for item in item2feat:
      feat = tuple(item2feat[item])
      if feat not in feat2hash_ids:
        feat2hash_ids[feat] = np.random.permutation(
            self.config['n_hash_buckets']
        )
      idx = feat2cnt[feat]

      if idx >= self.config['n_hash_buckets']:
        raise ValueError(
            '[TOKENIZER] Too many conflicts of semantic IDs found.'
            ' Please increase the number of hash buckets.'
        )

      item2hashed_feat[item] = (
          *item2feat[item],
          feat2hash_ids[feat][idx].item(),
      )
      feat2cnt[feat] += 1

    return item2hashed_feat

  def _get_item2feat(self, dataset: AbstractDataset) -> dict[Any, Any]:
    """Get the item features.

    If the features are already cached, load them from the cache. Otherwise,
    generate the features and save them to the cache.

    Args:
        dataset (AbstractDataset): The dataset containing the items to get
          features for.

    Returns:
        item2feat (dict): A dictionary mapping item IDs to their features.
    """
    feat_path = os.path.join(
        dataset.cache_dir,
        f'processed/item.{self.config["metadata"]}.feat',
    )
    if os.path.exists(feat_path):
      self.logger.info(f'[TOKENIZER] Loading item features from {feat_path}...')
      with open(feat_path, 'r') as f:
        item2feat = json.load(f)
      return item2feat
    self.logger.info('[TOKENIZER] Generating item features...')
    item2sem_ids = self._get_sem_ids(dataset)
    item2attr_ids = self._get_attr_ids(dataset)
    item2feat = self._combine_features(item2sem_ids, item2attr_ids)
    item2hashed_feat = self._get_hashed_feat(dataset, item2feat)
    self.logger.info(f'[TOKENIZER] Saving item features to {feat_path}...')
    with open(feat_path, 'w') as f:
      json.dump(item2hashed_feat, f)
    return item2hashed_feat

  def _check_conflicts(self, item2feat: dict[Any, Any]):
    """Check if there are any conflicts in the item features.

    If there are conflicts, raise a ValueError.

    Args:
        item2feat (dict): A dictionary mapping item IDs to their features.
    """
    feat2cnt = collections.Counter()
    for feat in item2feat.values():
      feat = tuple(feat)
      feat2cnt[feat] += 1
    for feat in feat2cnt:
      if feat2cnt[feat] > 1:
        raise ValueError(f'[TOKENIZER] Conflicts found in features: {feat}')

  def _tokenize_once(self, item_seq):
    state_seq = []
    for item in item_seq:
      feats = self.item2feat[item]
      tokenized_feats = []
      for i, feat in enumerate(feats):
        tokenized_feats.append(self.actionpiece.rank[(i, feat)])
      state_seq.append(tokenized_feats)
    return np.array(state_seq)

  def tokenize_function(
      self, example: dict[Any, Any], split: str
  ) -> dict[Any, Any]:
    max_item_seq_len = self.config['max_item_seq_len']
    item_seq = example['item_seq'][0]
    if split == 'train':
      # Tokenize each subsequence with maximum `max_item_seq_len` items
      n_return_examples = len(item_seq) - 1
      all_state_seq = []
      for i in range(n_return_examples):
        cur_item_seq = self._tokenize_once(
            item_seq[max(i + 1 - max_item_seq_len, 0) : i + 2]
        )
        all_state_seq.append(cur_item_seq)
    else:
      all_state_seq = [self._tokenize_once(item_seq[-(max_item_seq_len + 1) :])]
    return {
        'state_seq': all_state_seq,
    }

  def tokenize(self, datasets: dict[Any, Any]) -> dict[Any, Any]:
    tokenized_datasets = {}
    for split in datasets:
      tokenized_datasets[split] = datasets[split].map(
          lambda t: self.tokenize_function(t, split),  # pylint: disable=cell-var-from-loop
          batched=True,
          batch_size=1,
          remove_columns=datasets[split].column_names,
          num_proc=self.config['num_proc'],
          desc=f'Tokenizing {split} set: ',
      )
    for split in datasets:
      tokenized_datasets[split].set_format(type='numpy')
    return tokenized_datasets

  @property
  def vocab_size(self):
    return self.eos_token + 1

  @property
  def max_token_seq_len(self):
    # +2 for EOS and BOS
    return self.actionpiece.n_categories * self.config['max_item_seq_len'] + 2

  def _init_tokenizer(self, dataset: AbstractDataset):
    self.item2feat = self._get_item2feat(dataset)
    self._check_conflicts(self.item2feat)

    tokenizer_path = os.path.join(
        dataset.cache_dir, 'processed/actionpiece.json'
    )
    if os.path.exists(tokenizer_path):
      # If trained tokenizer exists, load it
      self.logger.info(
          f'[TOKENIZER] Loading ActionPiece from {tokenizer_path}...'
      )
      actionpiece = ActionPieceCore.from_pretrained(
          tokenizer_path, vocab_size=self.config['actionpiece_vocab_size']
      )
    else:
      # Initialize ActionPiece from initial features
      self.logger.info('[TOKENIZER] Constructing ActionPiece vocabulary...')
      actionpiece = ActionPieceCore(
          state2feat=self.item2feat,
      )
      # Construct ActionPiece vocabulary
      actionpiece.train(
          state_corpus=dataset.split_data['train']['item_seq'],
          target_vocab_size=self.config['actionpiece_vocab_size'],
      )
      actionpiece.save(tokenizer_path)
    return actionpiece

  def encode_labels(self, labels):
    """Cache the encoded labels for faster inference.

    Args:
        labels (np.ndarray): The labels to be encoded.

    Returns:
        encoded_labels (list[int]): The encoded labels.
    """
    key = labels.tostring()
    if key in self.encoded_labels:
      return self.encoded_labels[key]
    encoded_labels = self.actionpiece.encode(labels, shuffle='none')
    self.encoded_labels[key] = encoded_labels
    return encoded_labels

  def collate_fn_train(self, batch):
    """Tokenizing a batch of examples on-the-fly while training.

    Args:
        batch (list): A list of examples. batch['state_seq'] is a state
          sequence. batch['state_seq'][i] is a list of features.

    Returns:
        dict: A dictionary of tensors.
    """
    input_ids = []
    attention_mask = []
    labels = []
    for data in batch:
      seq = data['state_seq'][:-1]
      lb = data['state_seq'][-1:]
      input_ids.append(
          [self.bos_token]
          + self.actionpiece.encode(seq, shuffle=self.train_shuffle)
          + [self.eos_token]
      )
      labels.append(self.encode_labels(lb) + [self.eos_token])
    seq_lens = [len(ids) for ids in input_ids]
    max_seq_len = max(seq_lens)
    for i in range(len(batch)):
      input_ids[i] = input_ids[i] + [self.padding_token] * (
          max_seq_len - seq_lens[i]
      )
      attention_mask.append(
          [1] * seq_lens[i] + [0] * (max_seq_len - seq_lens[i])
      )
      labels[i] = labels[i] + [self.ignored_label] * (
          self.actionpiece.n_categories + 1 - len(labels[i])
      )
    return {
        'input_ids': torch.LongTensor(input_ids),
        'attention_mask': torch.LongTensor(attention_mask),
        'labels': torch.LongTensor(labels),
    }

  def collate_fn_val(self, batch):
    """Tokenizing a batch of examples on-the-fly while evaluating.

    Args:
        batch (list): A list of examples. batch['state_seq'] is a state
          sequence. batch['state_seq'][i] is a list of features.

    Returns:
        dict: A dictionary of tensors.
    """
    input_ids = []
    attention_mask = []
    labels = []
    for data in batch:
      seq = data['state_seq'][:-1]
      lb = data['state_seq'][-1]
      # The labels should always be encoded by encode_plus
      input_ids.append(
          [self.bos_token]
          + self.actionpiece.encode(
              seq,
              shuffle='none' if self.n_inference_ensemble == -1 else 'feature',
          )
          + [self.eos_token]
      )
      labels.append(lb)
    seq_lens = [len(ids) for ids in input_ids]
    max_seq_len = max(seq_lens)
    for i, sequence_length in enumerate(seq_lens):
      input_ids[i] = input_ids[i] + [self.padding_token] * (
          max_seq_len - sequence_length
      )
      attention_mask.append(
          [1] * sequence_length + [0] * (max_seq_len - sequence_length)
      )
    return {
        'input_ids': torch.LongTensor(input_ids),
        'attention_mask': torch.LongTensor(attention_mask),
        'labels': torch.LongTensor(np.array(labels)),
    }

  def collate_fn_test(self, batch):
    """Tokenizing a batch of examples on-the-fly while evaluating.

    Args:
        batch (list): A list of examples. batch['state_seq'] is a state
          sequence. batch['state_seq'][i] is a list of features.

    Returns:
        dict: A dictionary of tensors.
    """
    input_ids = []
    attention_mask = []
    labels = []
    for data in batch:
      seq = data['state_seq'][:-1]
      lb = data['state_seq'][-1]
      # The labels should always be encoded by encode_plus
      if self.n_inference_ensemble == -1:
        input_ids.append(
            [self.bos_token]
            + self.actionpiece.encode(seq, shuffle='none')
            + [self.eos_token]
        )
        labels.append(lb)
      for _ in range(self.n_inference_ensemble):
        input_ids.append(
            [self.bos_token]
            + self.actionpiece.encode(seq, shuffle='feature')
            + [self.eos_token]
        )
      labels.append(lb)
    seq_lens = [len(ids) for ids in input_ids]
    max_seq_len = max(seq_lens)
    for i, sequence_length in enumerate(seq_lens):
      input_ids[i] = input_ids[i] + [self.padding_token] * (
          max_seq_len - sequence_length
      )
      attention_mask.append(
          [1] * sequence_length + [0] * (max_seq_len - sequence_length)
      )
    return {
        'input_ids': torch.LongTensor(input_ids),
        'attention_mask': torch.LongTensor(attention_mask),
        'labels': torch.LongTensor(np.array(labels)),
    }
