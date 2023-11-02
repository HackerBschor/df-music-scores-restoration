import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
from skimage.util import random_noise
from skimage.io import imread, imshow


class ImageDestroyer:
	def __init__(self, input_file, output_file):
		self.input_file = input_file
		self.output_file = output_file

		self.im = imread(input_file)
		# self.img_width, self.img_height = self.im.size

	def rotate(self):
		expand = random.random() > 0.5
		rotation = random.randint(0, 10) if expand else random.randint(0, 3)
		rotation = (-1 if random.random() > 0.5 else 1) * rotation
		rotation = (180 if random.random() > 0.5 else 0) + rotation # Flip
		self.im = self.im.rotate(rotation, expand=expand)

	def blur(self):
		blurriness = random.random() # TODO: chances need like 0.1 of original value
		self.im = self.im.resize((int(blurriness * self.img_width), int(blurriness * self.img_height)))
		self.im = self.im.resize((self.img_width, self.img_height))

	def noisy(self):
		self.im = random_noise(self.im)

	def destroy_image(self):
		#self.rotate()
		#self.blur()
		self.noisy()
		imshow(self.im)
		plt.show()


if __name__ == '__main__':
	imd = ImageDestroyer("../dataset/pairs/perfect_big/Bach_Bwv001_400_Chorales_006707b_0.png", None)
	imd.destroy_image()
