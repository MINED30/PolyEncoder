import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os
from common_utils import pickle_load, pickle_dump

# 데이터셋 로드
class SelectionDataset(Dataset):
  def __init__(self, file_path, context_transform, response_transform, sample_cnt=None):
    self.context_transform = context_transform # SelectionJoinTransform을 불러옴 (데이터셋>>토크나이징>>input_ids_list, segment_ids_list, input_masks_list)
    self.response_transform = response_transform # SelectionJoinTransform을 불러옴 (데이터셋>>토크나이징>>input_ids_list, segment_ids_list, input_masks_list)

    self.data_source = [] # 데이터 소스 담을 리스트 생성
    self.transformed_data = {} # transformed된 데이터담을 딕셔너리 생성

    cache_path = file_path + "_" + str(context_transform) + '_samplecnt%s' % str(sample_cnt) + '.cache'
    if os.path.exists(cache_path):
      # 캐싱해서 기존에 했던거 찾아냄
      self.transformed_data = pickle_load(cache_path)
      self.data_source = [0] * len(self.transformed_data)
    else:
      # 아니면 새로 만들음
      with open(file_path, encoding='utf-8') as f:
        group = {
          'context': None,
          'responses': [],
          'labels': []
        }
        for line in f:
          # 라인별로 라벨, context, response로 나눔
          split = line.strip().split('\t')
          lbl, context, response = int(split[0]), split[1:-1], split[-1]
          if lbl == 1 and len(group['responses']) > 0:
            self.data_source.append(group)
            group = {
              'context': None,
              'responses': [],
              'labels': []
            }
            if sample_cnt is not None and len(self.data_source) >= sample_cnt:
              break
          group['responses'].append(response)
          group['labels'].append(lbl)
          group['context'] = context
        if len(group['responses']) > 0:
          self.data_source.append(group)
      
      # progress bar 용
      for idx in tqdm(range(len(self.data_source))):
        self.__get_single_item__(idx)
      
      # transform된거 캐시패쓰에 덤프함
      pickle_dump(self.transformed_data, cache_path)
      self.data_source = [0] * len(self.transformed_data)

  def __len__(self):
    return len(self.data_source)

  # getter역할
  def __getitem__(self, indices):
    if isinstance(indices, (tuple, list)):
      return [self.__get_single_item__(index) for index in indices]
    return self.__get_single_item__(indices)

  def __get_single_item__(self, index):
    if index in self.transformed_data:
      key_data = self.transformed_data[index]
      return key_data
    else:
      group = self.data_source[index]
      context, responses, labels = group['context'], group['responses'], group['labels']
      transformed_context = self.context_transform(context)  # [token_ids],[seg_ids],[masks]
      transformed_responses = self.response_transform(responses)  # [token_ids],[seg_ids],[masks]
      key_data = transformed_context, transformed_responses, labels
      self.transformed_data[index] = key_data

      return key_data

  # 배치화하는 작업 (배치별 token ids, segment ids, mask)
  def batchify(self, batch):
    contexts_token_ids_list_batch, contexts_segment_ids_list_batch, contexts_input_masks_list_batch, contexts_masks_batch, \
    responses_token_ids_list_batch, responses_segment_ids_list_batch, responses_input_masks_list_batch = [], [], [], [], [], [], []
    labels_batch = []
    for sample in batch:
      (contexts_token_ids_list, contexts_segment_ids_list, contexts_input_masks_list, contexts_masks_list), \
      (responses_token_ids_list, responses_segment_ids_list, responses_input_masks_list, _) = sample[:2]

      contexts_token_ids_list_batch.append(contexts_token_ids_list)
      contexts_segment_ids_list_batch.append(contexts_segment_ids_list)
      contexts_input_masks_list_batch.append(contexts_input_masks_list)
      contexts_masks_batch.append(contexts_masks_list)

      responses_token_ids_list_batch.append(responses_token_ids_list)
      responses_segment_ids_list_batch.append(responses_segment_ids_list)
      responses_input_masks_list_batch.append(responses_input_masks_list)

      labels_batch.append(sample[-1])

    long_tensors = [contexts_token_ids_list_batch, contexts_segment_ids_list_batch, contexts_input_masks_list_batch,
                    contexts_masks_batch,
                    responses_token_ids_list_batch, responses_segment_ids_list_batch, responses_input_masks_list_batch]

    contexts_token_ids_list_batch, contexts_segment_ids_list_batch, contexts_input_masks_list_batch, contexts_masks_batch, \
    responses_token_ids_list_batch, responses_segment_ids_list_batch, responses_input_masks_list_batch = (
      torch.tensor(t, dtype=torch.long) for t in long_tensors)

    labels_batch = torch.tensor(labels_batch, dtype=torch.long)
    return contexts_token_ids_list_batch, contexts_segment_ids_list_batch, contexts_input_masks_list_batch, contexts_masks_batch, \
           responses_token_ids_list_batch, responses_segment_ids_list_batch, responses_input_masks_list_batch, labels_batch

  def batchify_join_str(self, batch):
    contexts_token_ids_list_batch, contexts_segment_ids_list_batch, contexts_input_masks_list_batch, \
    responses_token_ids_list_batch, responses_segment_ids_list_batch, responses_input_masks_list_batch = [], [], [], [], [], []
    labels_batch = []
    for sample in batch:
      (contexts_token_ids_list, contexts_segment_ids_list, contexts_input_masks_list), \
      (responses_token_ids_list, responses_segment_ids_list, responses_input_masks_list, _) = sample[:2]

      contexts_token_ids_list_batch.append(contexts_token_ids_list)
      contexts_segment_ids_list_batch.append(contexts_segment_ids_list)
      contexts_input_masks_list_batch.append(contexts_input_masks_list)

      responses_token_ids_list_batch.append(responses_token_ids_list)
      responses_segment_ids_list_batch.append(responses_segment_ids_list)
      responses_input_masks_list_batch.append(responses_input_masks_list)

      labels_batch.append(sample[-1])

    long_tensors = [contexts_token_ids_list_batch, contexts_segment_ids_list_batch, contexts_input_masks_list_batch,
                    responses_token_ids_list_batch, responses_segment_ids_list_batch, responses_input_masks_list_batch]

    contexts_token_ids_list_batch, contexts_segment_ids_list_batch, contexts_input_masks_list_batch, \
    responses_token_ids_list_batch, responses_segment_ids_list_batch, responses_input_masks_list_batch = (
      torch.tensor(t, dtype=torch.long) for t in long_tensors)

    labels_batch = torch.tensor(labels_batch, dtype=torch.long)
    return contexts_token_ids_list_batch, contexts_segment_ids_list_batch, contexts_input_masks_list_batch, \
           responses_token_ids_list_batch, responses_segment_ids_list_batch, responses_input_masks_list_batch, labels_batch