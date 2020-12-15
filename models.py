from  tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Input, Concatenate, Lambda

def center_branch_simple_model(dim_left, dim_right, dim_dense = 2048, activation = "relu", bias = False):
  center_input = Input(shape=(dim_left + dim_right,), name='center_input')

  center_left = Lambda(lambda x: x[:,:dim_left])(center_input)
  center_right = Lambda(lambda x: x[:,-dim_right:])(center_input)

  left_branch = Dense(dim_dense, input_dim=4096, name='left_branch', activation=activation, use_bias = bias)(center_left)

  right_branch = Dense(dim_dense, input_dim=768, name='right_branch', activation=activation, use_bias = bias)(center_right)

  salida = Concatenate(axis=1, name = 'salida')([left_branch, right_branch])

  return Model(inputs=[center_input], outputs=salida)

# Model with a extra branch to concatenate layer
def center_branch_simple_model_3(dim_left, dim_right, dim_dense = 2048, activation = "relu", bias = False):
  center_input = Input(shape=(dim_left + dim_right), name='center_input')

  center_left = Lambda(lambda x: x[:,:dim_left])(center_input)
  center_right = Lambda(lambda x: x[:,-dim_right:])(center_input)

  left_branch = Dense(dim_dense, input_dim=4096, name='left_branch', activation=activation, use_bias = bias)(center_left)

  right_branch = Dense(dim_dense, input_dim=768, name='right_branch', activation=activation, use_bias = bias)(center_right)

  salida = Concatenate(axis=1, name = 'salida')([left_branch, right_branch, center_right])

  return Model(inputs=[center_input], outputs=salida)

def center_branch_augmented_model_3(dim_left, dim_right, dim_dense = 2048, activation = "sigmoid", bias = False):
  center_input = Input(shape=(dim_left + dim_right), name='center_input')

  center_left = Lambda(lambda x: x[:,:dim_left])(center_input)
  center_right = Lambda(lambda x: x[:,-dim_right:])(center_input)

  left_linear = Dense(dim_dense, input_dim=4096, name='left_linear', activation=None, use_bias = bias)(center_left) # linear
  left_sigma = Dense(dim_dense, name='left_sigma', activation=activation, use_bias = bias)(left_linear) # activation
  left_branch = Multiply(name='left_branch')([left_linear, left_sigma]) # hadamard

  right_linear = Dense(dim_dense, input_dim=768, name='right_linear', activation=None, use_bias = bias)(center_right) # linear
  right_sigma = Dense(dim_dense, name='right_sigma', activation=activation, use_bias = bias)(right_linear) # activation
  right_branch = Multiply(name='right_branch')([right_linear, right_sigma]) # hadamard

  salida = Concatenate(axis=1, name = 'salida')([left_branch, right_branch, center_right])

  return Model(inputs=[center_input], outputs=salida)