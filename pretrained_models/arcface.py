from collections import namedtuple
import torch
from torch.nn import Conv2d, BatchNorm2d, PReLU, ReLU, Sigmoid, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Linear, Dropout, BatchNorm1d
import torch.nn.functional as F
"""
Modified Backbone implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
"""


class Flatten(Module):
	def forward(self, input):
		return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
	norm = torch.norm(input, 2, axis, True)
	output = torch.div(input, norm)
	return output


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
	""" A named tuple describing a ResNet block. """


def get_block(in_channel, depth, num_units, stride=2):
	return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
	if num_layers == 50:
		blocks = [
			get_block(in_channel=64, depth=64, num_units=3),
			get_block(in_channel=64, depth=128, num_units=4),
			get_block(in_channel=128, depth=256, num_units=14),
			get_block(in_channel=256, depth=512, num_units=3)
		]
	elif num_layers == 100:
		blocks = [
			get_block(in_channel=64, depth=64, num_units=3),
			get_block(in_channel=64, depth=128, num_units=13),
			get_block(in_channel=128, depth=256, num_units=30),
			get_block(in_channel=256, depth=512, num_units=3)
		]
	elif num_layers == 152:
		blocks = [
			get_block(in_channel=64, depth=64, num_units=3),
			get_block(in_channel=64, depth=128, num_units=8),
			get_block(in_channel=128, depth=256, num_units=36),
			get_block(in_channel=256, depth=512, num_units=3)
		]
	else:
		raise ValueError("Invalid number of layers: {}. Must be one of [50, 100, 152]".format(num_layers))
	return blocks


class SEModule(Module):
	def __init__(self, channels, reduction):
		super(SEModule, self).__init__()
		self.avg_pool = AdaptiveAvgPool2d(1)
		self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
		self.relu = ReLU(inplace=True)
		self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
		self.sigmoid = Sigmoid()

	def forward(self, x):
		module_input = x
		x = self.avg_pool(x)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.sigmoid(x)
		return module_input * x


class bottleneck_IR(Module):
	def __init__(self, in_channel, depth, stride):
		super(bottleneck_IR, self).__init__()
		if in_channel == depth:
			self.shortcut_layer = MaxPool2d(1, stride)
		else:
			self.shortcut_layer = Sequential(
				Conv2d(in_channel, depth, (1, 1), stride, bias=False),
				BatchNorm2d(depth)
			)
		self.res_layer = Sequential(
			BatchNorm2d(in_channel),
			Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
			Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth)
		)

	def forward(self, x):
		shortcut = self.shortcut_layer(x)
		res = self.res_layer(x)
		return res + shortcut


class bottleneck_IR_SE(Module):
	def __init__(self, in_channel, depth, stride):
		super(bottleneck_IR_SE, self).__init__()
		if in_channel == depth:
			self.shortcut_layer = MaxPool2d(1, stride)
		else:
			self.shortcut_layer = Sequential(
				Conv2d(in_channel, depth, (1, 1), stride, bias=False),
				BatchNorm2d(depth)
			)
		self.res_layer = Sequential(
			BatchNorm2d(in_channel),
			Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
			PReLU(depth),
			Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
			BatchNorm2d(depth),
			SEModule(depth, 16)
		)

	def forward(self, x):
		shortcut = self.shortcut_layer(x)
		res = self.res_layer(x)
		return res + shortcut


class SeparableConv2d(torch.nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size, bias=False):
		super(SeparableConv2d, self).__init__()
		self.depthwise = Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, bias=bias, padding=1)
		self.pointwise = Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

	def forward(self, x):
		out = self.depthwise(x)
		out = self.pointwise(out)
		return out


def _upsample_add(x, y):
	"""Upsample and add two feature maps.
	Args:
	  x: (Variable) top feature map to be upsampled.
	  y: (Variable) lateral feature map.
	Returns:
	  (Variable) added feature map.
	Note in PyTorch, when input size is odd, the upsampled feature map
	with `F.upsample(..., scale_factor=2, mode='nearest')`
	maybe not equal to the lateral feature map size.
	e.g.
	original input size: [N,_,15,15] ->
	conv2d feature map size: [N,_,8,8] ->
	upsampled feature map size: [N,_,16,16]
	So we choose bilinear upsample which supports arbitrary output sizes.
	"""
	_, _, H, W = y.size()
	return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y


class SeparableBlock(Module):

	def __init__(self, input_size, kernel_channels_in, kernel_channels_out, kernel_size):
		super(SeparableBlock, self).__init__()

		self.input_size = input_size
		self.kernel_size = kernel_size
		self.kernel_channels_in = kernel_channels_in
		self.kernel_channels_out = kernel_channels_out

		self.make_kernel_in  = Linear(input_size, kernel_size * kernel_size * kernel_channels_in)
		self.make_kernel_out = Linear(input_size, kernel_size * kernel_size * kernel_channels_out)

		self.kernel_linear_in = Linear(kernel_channels_in, kernel_channels_in)
		self.kernel_linear_out = Linear(kernel_channels_out, kernel_channels_out)

	def forward(self, features):

		features = features.view(-1, self.input_size)

		kernel_in = self.make_kernel_in(features).view(-1, self.kernel_size, self.kernel_size, 1, self.kernel_channels_in)
		kernel_out = self.make_kernel_out(features).view(-1, self.kernel_size, self.kernel_size, self.kernel_channels_out, 1)

		kernel = torch.matmul(kernel_out, kernel_in)

		kernel = self.kernel_linear_in(kernel).permute(0, 1, 2, 4, 3)
		kernel = self.kernel_linear_out(kernel)
		kernel = kernel.permute(0, 4, 3, 1, 2)

		return kernel


class Backbone(Module):
	def __init__(self, input_size, num_layers, mode='ir', drop_ratio=0.4, affine=True):
		super(Backbone, self).__init__()
		assert input_size in [112, 224], "input_size should be 112 or 224"
		assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
		assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
		blocks = get_blocks(num_layers)
		if mode == 'ir':
			unit_module = bottleneck_IR
		elif mode == 'ir_se':
			unit_module = bottleneck_IR_SE
		self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
									  BatchNorm2d(64),
									  PReLU(64))
		if input_size == 112:
			self.output_layer = Sequential(BatchNorm2d(512),
			                               Dropout(drop_ratio),
			                               Flatten(),
			                               Linear(512 * 7 * 7, 512),
			                               BatchNorm1d(512, affine=affine))
		else:
			self.output_layer = Sequential(BatchNorm2d(512),
			                               Dropout(drop_ratio),
			                               Flatten(),
			                               Linear(512 * 14 * 14, 512),
			                               BatchNorm1d(512, affine=affine))

		modules = []
		for block in blocks:
			for bottleneck in block:
				modules.append(unit_module(bottleneck.in_channel,
										   bottleneck.depth,
										   bottleneck.stride))
		self.body = Sequential(*modules)

	def forward(self, x):
		x = self.input_layer(x)
		x = self.body(x)
		x = self.output_layer(x)
		return l2_norm(x)