import torch.nn.functional as F
from torch import nn


class Denoiser(nn.Module):

	def __init__(self):
		super().__init__();
		# defining the encoder
		self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

		self.pool = nn.MaxPool2d(2, 2)

		# defining the decoder
		self.convt_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
		self.convt_2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
		self.convt_3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

	def forward(self, x):
		# passing the image through encoder
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))

		# passing the encoded part through decoder
		x = F.relu(self.convt_1(x))
		x = F.interpolate(x, scale_factor=2, mode='nearest')
		x = F.relu(self.convt_2(x))
		x = F.interpolate(x, scale_factor=2, mode='nearest')
		x = F.sigmoid(self.convt_3(x))

		return x
