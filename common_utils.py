import pickle


def pickle_dump(data, file_path):
  ''' 
  Pickle에 저장
  '''
  f_write = open(file_path, 'wb')
  pickle.dump(data, f_write, True)


def pickle_load(file_path):
  ''' 
  Pickle에서 로드
  '''
  f_read = open(file_path, 'rb')
  data = pickle.load(f_read)

  return data
