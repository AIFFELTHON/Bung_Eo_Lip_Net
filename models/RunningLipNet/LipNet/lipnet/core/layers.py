from keras.layers.core import Lambda  # 람다식을 Layer 객체로 매핑
from lipnet.core.loss import ctc_lambda_func

# CTC Layer implementation using Lambda layer
# (because Keras doesn't support extra prams on loss function)
def CTC(name, args):
	return Lambda(ctc_lambda_func, output_shape=(1,), name=name)(args)