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

"""Pipeline for ActionPiece."""

import logging
import os
from typing import Any

import accelerate as accelerate_lib
from genrec import utils
# 移除直接导入，改为延迟导入
# from genrec.dataset import AbstractDataset
# from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
import torch
from torch.utils import data


DataLoader = data.DataLoader


class Pipeline:
  """Pipeline for ActionPiece.

  This class orchestrates the training and evaluation of an ActionPiece model,
  including:
    - Loading and configuring the dataset.
    - Initializing the tokenizer.
    - Setting up the model.
    - Creating the trainer.
    - Preparing data loaders.
    - Running the training and evaluation loop.

  Attributes:
      config: A dictionary containing the configuration parameters.
      project_dir: The directory for the accelerator.
      accelerator: An instance of the accelerate.Accelerator class.
      logger: An instance of the logging.Logger class.
      raw_dataset: The raw dataset instance.
      split_datasets: A dictionary containing the split datasets (train, val,
        test).
      tokenizer: The tokenizer instance.
      tokenized_datasets: A dictionary containing the tokenized datasets.
      model: The model instance.
      trainer: The trainer instance.
  """

  def __init__(
      self,
      model_name: str | Any,  # 改为 Any 以避免循环导入
      dataset_name: str | Any,  # 改为 Any 以避免循环导入
      tokenizer: AbstractTokenizer | None = None,
      trainer=None,
      config_dict: dict[str, Any] | None = None,
      config_file: str = None,
  ):
    self.config = utils.get_config(
        model_name=model_name,
        dataset_name=dataset_name,
        config_file=config_file,
        config_dict=config_dict,
    )
    
    # 调试：打印可能有问题的配置项
    print(f"DEBUG: tensorboard_log_dir = {self.config.get('tensorboard_log_dir')} (type: {type(self.config.get('tensorboard_log_dir'))})")
    print(f"DEBUG: dataset = {self.config.get('dataset')} (type: {type(self.config.get('dataset'))})")
    print(f"DEBUG: model = {self.config.get('model')} (type: {type(self.config.get('model'))})")
    
    # Automatically set devices and ddp
    self.config['device'], self.config['use_ddp'] = utils.init_device()

    # Accelerator
    self.config['accelerator'] = accelerate_lib.Accelerator(
        log_with='tensorboard', project_dir=self.project_dir
    )

    # Seed and Logger
    utils.init_seed(self.config['rand_seed'], self.config['reproducibility'])
    utils.init_logger(self.config)
    self.logger = logging.getLogger()
    self.log(f'Device: {self.config["device"]}')

    # Dataset
    self.raw_dataset = utils.get_dataset(dataset_name)(self.config)
    self.log(self.raw_dataset)
    self.split_datasets = self.raw_dataset.split()

    # Tokenizer
    if tokenizer is not None:
      self.tokenizer = tokenizer(self.config, self.raw_dataset)
    else:
      assert isinstance(
          model_name, str
      ), 'Tokenizer must be provided if model_name is not a string.'
      self.tokenizer = utils.get_tokenizer(model_name)(
          self.config, self.raw_dataset
      )
    self.tokenized_datasets = self.tokenizer.tokenize(self.split_datasets)

    # Model
    with self.config['accelerator'].main_process_first():
      self.model = utils.get_model(model_name)(
          self.config, self.raw_dataset, self.tokenizer
      )
    self.log(self.model)
    self.log(self.model.n_parameters)

    # Trainer
    if trainer is not None:
      self.trainer = trainer
    else:
      self.trainer = utils.get_trainer(model_name)(
          self.config, self.model, self.tokenizer
      )

  @property
  def project_dir(self) -> str:
    """Returns the directory for the accelerator."""
    # 辅助函数：将任何值转换为字符串
    def to_string(value):
      if isinstance(value, list):
        if len(value) > 0:
          # 如果是列表，取第一个元素并转为字符串
          return str(value[0])
        else:
          return 'unknown'
      elif value is None:
        return 'unknown'
      else:
        return str(value)
    
    # 修复：确保所有路径组件都是字符串
    tensorboard_log_dir = to_string(self.config.get('tensorboard_log_dir', 'tensorboard'))
    dataset = to_string(self.config.get('dataset', 'unknown_dataset'))
    model = to_string(self.config.get('model', 'unknown_model'))
    
    return os.path.join(
        tensorboard_log_dir,
        dataset,
        model,
    )

  @property
  def accelerator(self) -> accelerate_lib.Accelerator:
    """Returns the accelerator instance."""
    return self.config['accelerator']

  def run(self):
    """Runs the training and evaluation pipeline.

    This method sets up data loaders, trains the model, evaluates it on the test
    set, and logs the results.
    """

    def get_dataloader(split, batch_size, shuffle):
      return DataLoader(
          self.tokenized_datasets[split],
          batch_size=batch_size,
          shuffle=shuffle,
          collate_fn=self.tokenizer.collate_fn[split],
      )
    # DataLoader
    train_dataloader = get_dataloader(
        'train', self.config['train_batch_size'], True
    )
    if self.config['n_inference_ensemble'] == -1:
      eval_batch_size = self.config['eval_batch_size']
    else:
      eval_batch_size = max(
          self.config['eval_batch_size'] // self.config['n_inference_ensemble'],
          1,
      )
    val_dataloader = get_dataloader('val', eval_batch_size, False)
    test_dataloader = get_dataloader('test', eval_batch_size, False)

    self.trainer.fit(train_dataloader, val_dataloader)

    self.accelerator.wait_for_everyone()
    self.model = self.accelerator.unwrap_model(self.model)

    self.model.load_state_dict(torch.load(self.trainer.saved_model_ckpt))
    self.model, test_dataloader = self.accelerator.prepare(
        self.model, test_dataloader
    )
    if self.accelerator.is_main_process:
      self.log(
          f'Loaded best model checkpoint from {self.trainer.saved_model_ckpt}'
      )
    test_results = self.trainer.evaluate(test_dataloader)

    if self.accelerator.is_main_process:
      for key in test_results:
        self.trainer.accelerator.log({f'Test_Metric/{key}': test_results[key]})
    self.log(f'Test Results: {test_results}')

    self.trainer.end()

  def log(self, message, level='info'):
    return utils.log(
        message, self.config['accelerator'], self.logger, level=level
    )