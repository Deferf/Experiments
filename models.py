from  tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Input, Concatenate, Lambda

def center_branch_simple_model(dim_left, dim_right, dim_dense = 2048):
  center_input = Input(shape=(dim_left + dim_right,), name='center_input')

  center_left = Lambda(lambda x: x[:,:dim_left])(center_input)
  center_right = Lambda(lambda x: x[:,-dim_right:])(center_input)

  left_branch = Dense(dim_dense, input_dim=4096, name='left_branch', activation='relu')(center_left)

  right_branch = Dense(dim_dense, input_dim=768, name='right_branch', activation='relu')(center_right)

  salida = Concatenate(axis=1, name = 'salida')([left_branch, right_branch])

  return Model(inputs=[center_input], outputs=salida)