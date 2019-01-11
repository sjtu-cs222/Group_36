import torch
from torch.autograd import Variable
from torchvision import models
import sys
import numpy as np
def replace_layers(model, i, indexes, layers):
	if i in indexes:
		return layers[indexes.index(i)]
	return model[i]

def prune_vgg16_conv_layer(model, layer_index, filter_index):
	_, conv = list(model.features._modules.items())[layer_index]
	  
	old_weights = conv.weight.data.cpu().numpy()
	  
	# new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
	# new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]

	### YOU CODE HERE ###
	target_filter = old_weights[filter_index, : , :, :]
	# max_value = np.max(old_weights[filter_index, : , :, :])
	# min_value = np.min(old_weights[filter_index, : , :, :])
	max_value = 0.05
	min_value = -0.05
	new_filter = np.random.rand(target_filter.shape[0],target_filter.shape[1],target_filter.shape[2]) * (max_value - min_value) + min_value
	# new_filter = np.zeros(target_filter.shape)
	old_weights[filter_index,:,:,:] = new_filter
	# new_weights[:, filter_index, : , :] = new_filter
	conv.weight.data = torch.from_numpy(old_weights).cuda()
	### YOU DO NOT CODE HERE

	bias_numpy = conv.bias.data.cpu().numpy()

	bias = np.zeros(shape = (bias_numpy.shape[0]), dtype = np.float32)
	bias[:filter_index] = bias_numpy[:filter_index]
	bias[filter_index] = 0
	bias[filter_index+1 : ] = bias_numpy[filter_index + 1 :]
	conv.bias.data = torch.from_numpy(bias).cuda()
	return model

if __name__ == '__main__':
	model = models.vgg16(pretrained=True)
	model.train()

	t0 = time.time()
	model = prune_conv_layer(model, 28, 10)
	print("The prunning took", time.time() - t0)