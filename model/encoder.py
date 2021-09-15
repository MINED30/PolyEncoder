import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, DistilBertModel
from model import dot_attention

# BI-ENCODER
class BertDssmModel(BertPreTrainedModel):
  def __init__(self, config, *inputs, **kwargs):
    # 사전훈련된 버트모델 그대로 상속받음
    super().__init__(config, *inputs, **kwargs)
    self.bert = kwargs['bert']
    try:
      self.dropout = nn.Dropout(config.hidden_dropout_prob)
      self.context_fc = nn.Linear(config.hidden_size, 64)
      self.response_fc = nn.Linear(config.hidden_size, 64)
    except:
      self.dropout = nn.Dropout(config.dropout)
      self.context_fc = nn.Linear(config.dim, 64)
      self.response_fc = nn.Linear(config.dim, 64)

  def forward(self, context_input_ids, context_segment_ids, context_input_masks,
              responses_input_ids, responses_segment_ids, responses_input_masks, labels=None):
    '''
    input_ids, segment_ids, input_masks를 context와 response로 부터 받으면

    '''
    ## only select the first response (whose lbl==1)
    if labels is not None:
      responses_input_ids = responses_input_ids[:, 0, :].unsqueeze(1)
      responses_segment_ids = responses_segment_ids[:, 0, :].unsqueeze(1)
      responses_input_masks = responses_input_masks[:, 0, :].unsqueeze(1)

    ##### CONTEXT VECTOR 뽑기
    # DistilBERT로 불러왔으면 
    if isinstance(self.bert, DistilBertModel):
      context_vec = self.bert(context_input_ids, context_input_masks)[-1]  # [bs,dim], segment가 안붙음
      context_vec = context_vec[:, 0]
    # 그냥버트면
    else:
      context_vec = self.bert(context_input_ids, context_input_masks, context_segment_ids)[-1]  # [bs,dim]

    ##### RESPONSE VECTOR 뽑기
    batch_size, res_cnt, seq_length = responses_input_ids.shape
    responses_input_ids = responses_input_ids.view(-1, seq_length)
    responses_input_masks = responses_input_masks.view(-1, seq_length)
    responses_segment_ids = responses_segment_ids.view(-1, seq_length)
    # DistilBERT로 불러왔으면 
    if isinstance(self.bert, DistilBertModel):
      responses_vec = self.bert(responses_input_ids, responses_input_masks)[-1]  # [bs,dim]
      responses_vec = responses_vec[:, 0]
    # 그냥버트면
    else:
      responses_vec = self.bert(responses_input_ids, responses_input_masks, responses_segment_ids)[-1]  # [bs,dim]
    responses_vec = responses_vec.view(batch_size, res_cnt, -1)


    # 갖고있는 Context vector와 Response vector를 가지고
    context_vec = self.context_fc(self.dropout(context_vec)) # dropout 시키고 fc레이어를 통과시켜서
    context_vec = F.normalize(context_vec, 2, -1) #nomalize한다음에
    responses_vec = self.response_fc(self.dropout(responses_vec)) # dropout 시키고 fc레이어를 통과시켜서
    responses_vec = F.normalize(responses_vec, 2, -1) #nomalize한다음에

    # label이 있는경우 (훈련시)
    if labels is not None:
      responses_vec = responses_vec.squeeze(1)
      dot_product = torch.matmul(context_vec, responses_vec.t())  # [bs, bs], context vector와 response vecotr를 그냥 matmul시켜줌 (matmul 시키기위해 response vector는 전치)
      mask = torch.eye(context_input_ids.size(0)).to(context_input_ids.device) # torch.eye는 [[1,0,0],[0,1,0],[0,0,1]] 같이 대각행렬만드는 메쏘드, positive인것만 얼마나 잘맞췄나 보는거 >>  device로
      loss = F.log_softmax(dot_product * 5, dim=-1) * mask # 5를 왜곱할까요? softmax를 취해줘서 loss를 구합니다
      loss = (-loss.sum(dim=1)).mean() # loss는 다합쳐서 평균으로 , 로스는 작아지게 하므로 음수값으로 구함(다맞춘 경우가 -1임)
      return loss

    else:
      # label이 없는경우 (예측시)
      context_vec = context_vec.unsqueeze(1)
      dot_product = torch.matmul(context_vec, responses_vec.permute(0, 2, 1))  # take this as logits
      dot_product.squeeze_(1)
      cos_similarity = (dot_product + 1) / 2
      return cos_similarity


# POLY-ENCODER
class BertPolyDssmModel(BertPreTrainedModel):
  def __init__(self, config, *inputs, **kwargs):
    # 사전훈련된 버트모델 그대로 상속받음
    super().__init__(config, *inputs, **kwargs)
    self.bert = kwargs['bert']
    self.vec_dim = 64
    self.poly_m = kwargs['poly_m'] # poly encoder에서 코드 몇개줄지 결정 (default=16)
    self.poly_code_embeddings = nn.Embedding(self.poly_m + 1, config.hidden_size) # 코드 갯수대로 가중치 행렬 생성 (왜 +1하지? bias인가?)
    try:
      self.dropout = nn.Dropout(config.hidden_dropout_prob)
      self.context_fc = nn.Linear(config.hidden_size, self.vec_dim)
      self.response_fc = nn.Linear(config.hidden_size, self.vec_dim)
    except:
      self.dropout = nn.Dropout(config.dropout)
      self.context_fc = nn.Linear(config.dim, self.vec_dim)
      self.response_fc = nn.Linear(config.dim, self.vec_dim)

  def forward(self, context_input_ids, context_segment_ids, context_input_masks,
              responses_input_ids, responses_segment_ids, responses_input_masks, labels=None):
    
    # 라벨이 있는경우(훈련시)
    ## only select the first response (whose lbl==1)
    if labels is not None:
      responses_input_ids = responses_input_ids[:, 0, :].unsqueeze(1)
      responses_segment_ids = responses_segment_ids[:, 0, :].unsqueeze(1)
      responses_input_masks = responses_input_masks[:, 0, :].unsqueeze(1)
    batch_size, res_cnt, seq_length = responses_input_ids.shape #

    ##### CONTEXT VECTOR 뽑기
    # poly_code로 들어가기 전 context에서 벡터를 뽑아냄(이후 code 가중치끼리 곱할것임)
    if isinstance(self.bert, DistilBertModel):
      state_vecs = self.bert(context_input_ids, context_input_masks)[-1]  # [bs, length, dim]
    else:
      state_vecs = self.bert(context_input_ids, context_input_masks, context_segment_ids)[0]  # [bs, length, dim]
    
    # poly code를 불러와서
    poly_code_ids = torch.arange(self.poly_m, dtype=torch.long, device=context_input_ids.device)
    poly_code_ids += 1 # bias개념으로 하나 넣는건가..?
    poly_code_ids = poly_code_ids.unsqueeze(0).expand(batch_size, self.poly_m)
    poly_codes = self.poly_code_embeddings(poly_code_ids)

    # poly code랑 어텐션을 줌
    context_vecs = dot_attention(poly_codes, state_vecs, state_vecs, context_input_masks, self.dropout) # 쿼리(poly codes), 키 밸류 (context에서 뽑은 vector:state vector)

    ##### RESPONSE VECTOR 뽑기
    responses_input_ids = responses_input_ids.view(-1, seq_length)
    responses_input_masks = responses_input_masks.view(-1, seq_length)
    responses_segment_ids = responses_segment_ids.view(-1, seq_length)

    # poly_code로 들어가기 전 response에서 벡터를 뽑아냄
    if isinstance(self.bert, DistilBertModel):
      state_vecs = self.bert(responses_input_ids, responses_input_masks)[-1]  # [bs, length, dim]
    else:
      state_vecs = self.bert(responses_input_ids, responses_input_masks, responses_segment_ids)[0]  # [bs, length, dim]

    ######## 이하 4줄은 poly encoder랑 좀 안맞는 것같음. 왜 Response에서 또 poly code로 쿼리를 날리는지?
    ######## https://github.com/chijames/Poly-Encoder/blob/master/encoder.py에서는 이부분이 없네여 
    poly_code_ids = torch.zeros(batch_size * res_cnt, 1, dtype=torch.long, device=context_input_ids.device)
    poly_codes = self.poly_code_embeddings(poly_code_ids)
    responses_vec = dot_attention(poly_codes, state_vecs, state_vecs, responses_input_masks, self.dropout)
    responses_vec = responses_vec.view(batch_size, res_cnt, -1)

    context_vecs = self.context_fc(self.dropout(context_vecs)) # 여기서 context_vec는 poly로 임베딩된거임
    context_vecs = F.normalize(context_vecs, 2, -1)  # [bs, m, dim]
    responses_vec = self.response_fc(self.dropout(responses_vec)) #근데 여기도 poly로 임베딩된거임;; 수정필요해보임
    responses_vec = F.normalize(responses_vec, 2, -1)

    ## poly final context vector aggregation
    if labels is not None:
      responses_vec = responses_vec.view(1, batch_size, -1).expand(batch_size, batch_size, self.vec_dim)
    # response를 쿼리로, context를 키밸류로 어텐션
    final_context_vec = dot_attention(responses_vec, context_vecs, context_vecs, None, self.dropout)
    final_context_vec = F.normalize(final_context_vec, 2, -1)  # [bs, res_cnt, dim], res_cnt==bs when training

    # 이하 bi encoder와 동일
    dot_product = torch.sum(final_context_vec * responses_vec, -1)  # [bs, res_cnt], res_cnt==bs when training
    if labels is not None:
      mask = torch.eye(context_input_ids.size(0)).to(context_input_ids.device)
      loss = F.log_softmax(dot_product * 5, dim=-1) * mask
      loss = (-loss.sum(dim=1)).mean()

      return loss
    else:
      cos_similarity = (dot_product + 1) / 2
      return cos_similarity


# 이사람게 더 깔끔함
# https://github.com/chijames/Poly-Encoder/blob/master/encoder.py
class PolyEncoder(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = kwargs['bert']
        self.poly_m = kwargs['poly_m']
        self.poly_code_embeddings = nn.Embedding(self.poly_m, config.hidden_size)
        torch.nn.init.normal_(self.poly_code_embeddings.weight, config.hidden_size ** -0.5)

    # 그냥 dot attention
    def dot_attention(self, q, k, v):
        # q: [bs, poly_m, dim] or [bs, res_cnt, dim]
        # k=v: [bs, length, dim] or [bs, poly_m, dim]
        attn_weights = torch.matmul(q, k.transpose(2, 1)) # [bs, poly_m, length]
        attn_weights = F.softmax(attn_weights, -1)
        output = torch.matmul(attn_weights, v) # [bs, poly_m, dim]
        return output

    def forward(self, context_input_ids, context_input_masks,
                            responses_input_ids, responses_input_masks, labels=None):
        # input_ids랑 input_mask만가져옴
        # during training, only select the first response
        # we are using other instances in a batch as negative examples
        if labels is not None:
            responses_input_ids = responses_input_ids[:, 0, :].unsqueeze(1)
            responses_input_masks = responses_input_masks[:, 0, :].unsqueeze(1)
        batch_size, res_cnt, seq_length = responses_input_ids.shape # res_cnt is 1 during training

        # context 만 poly로 뽑아내서 임베딩시킴 (poly_m 만큼)
        # context encoder
        ctx_out = self.bert(context_input_ids, context_input_masks)[0]  # [bs, length, dim]
        poly_code_ids = torch.arange(self.poly_m, dtype=torch.long).to(context_input_ids.device)
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(batch_size, self.poly_m)
        poly_codes = self.poly_code_embeddings(poly_code_ids) # [bs, poly_m, dim]
        embs = self.dot_attention(poly_codes, ctx_out, ctx_out) # [bs, poly_m, dim]

        # response는 그냥 bert지나서 임베딩시킴
        # response encoder
        responses_input_ids = responses_input_ids.view(-1, seq_length)
        responses_input_masks = responses_input_masks.view(-1, seq_length)
        cand_emb = self.bert(responses_input_ids, responses_input_masks)[0][:,0,:] # [bs, dim]
        cand_emb = cand_emb.view(batch_size, res_cnt, -1) # [bs, res_cnt, dim]

        # merge
        if labels is not None:
            # we are recycling responses for faster training
            # we repeat responses for batch_size times to simulate test phase
            # so that every context is paired with batch_size responses
            cand_emb = cand_emb.permute(1, 0, 2) # [1, bs, dim]
            cand_emb = cand_emb.expand(batch_size, batch_size, cand_emb.shape[2]) # [bs, bs, dim]
            ctx_emb = self.dot_attention(cand_emb, embs, embs).squeeze() # [bs, bs, dim] # poly지난 context(embs)를 키, 밸류로하고 bert만 지난 response를 쿼리로 어텐션한 것이 최종 context vector가 됨
            dot_product = (ctx_emb*cand_emb).sum(-1) # [bs, bs] # 내적해서 구함
            mask = torch.eye(batch_size).to(context_input_ids.device) # [bs, bs] # 대각행렬로 나머지날리고
            loss = F.log_softmax(dot_product, dim=-1) * mask # 소프트맥스 취해서
            loss = (-loss.sum(dim=1)).mean() # 로스구함
            return loss
        else:
            ctx_emb = self.dot_attention(cand_emb, embs, embs) # [bs, res_cnt, dim]
            dot_product = (ctx_emb*cand_emb).sum(-1)
            return dot_product