import random
from random import shuffle

import matplotlib.pyplot as plt
import pandas as pd

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np

if __name__ == '__main__':
    random.seed(2)

    index_file = "../dataset/pairs/perfect_broken_index.pkl"
    path_broken = "../dataset/pairs/broken"
    path_perfect = "../dataset/pairs/perfect"

    df = pd.read_pickle(index_file)
    df.head(5)

    transform = transforms.Compose(
        [transforms.Resize((3508, 2480)), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])


    import sys

    max_images = 500

    images_broken_perfect = []

    for file_perfect in df.index:
        # print("\r", "{:02d}".format(max_images), end="")

        if max_images == 0:
            break

        files_broken = df.loc[file_perfect, "file_broken"]

        if files_broken is None:
            continue

        file_broken = files_broken.values[random.randint(0, len(files_broken.values) - 1)]  # Random "Broken" image

        image_broken = Image.open(f"{path_broken}/{file_broken}").convert('L')
        image_perfect = Image.open(f"{path_perfect}/{file_perfect}").convert('L')

        image_broken = transform(image_broken)
        image_perfect = transform(image_perfect)

        print(sys.getsizeof(image_broken.storage()))

        images_broken_perfect.append((image_broken, image_perfect))

        max_images -= 1

    print("\rDone")

    def revoke_preprocessing(image: [torch.Tensor, np.ndarray]):
        if type(image) == torch.Tensor:
            image = image.squeeze().numpy()

        image = (0.5 * image + 0.5) * 255
        return Image.fromarray(image).convert('RGB')


    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(revoke_preprocessing(images_broken_perfect[0][0]))
    ax1.set_title("Broken")
    ax2.imshow(revoke_preprocessing(images_broken_perfect[0][1]))
    ax2.set_title("Perfect")
    plt.show()

    train_test_val = (0.8, 0.1, 0.1)
    limits = [int(x * len(images_broken_perfect)) for x in train_test_val]

    shuffle(images_broken_perfect)

    train_dataset = images_broken_perfect[:limits[0]]
    test_dataset = images_broken_perfect[limits[0]:limits[0] + limits[1]]
    valid_dataset = images_broken_perfect[limits[0] + limits[1]:]

    print("Sizes: train", len(train_dataset), ", test", len(test_dataset), ", validation", len(valid_dataset))
    # %%
    batch_size = 4

    train_loader = DataLoader(train_dataset)
    test_loader = DataLoader(test_dataset)
    valid_loader = DataLoader(valid_dataset)
    # %% md
    # Train the model
    # %%
    import torch.nn.functional as F


    class Denoiser(nn.Module):
        def __init__(self):
            super().__init__()
            # defining the encoder
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)

            # defining pooling
            self.pool = nn.MaxPool2d(2, 2)

            # defining the decoder
            self.conv2d_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
            self.conv2d_2 = nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1)
            self.conv2d_3 = nn.Conv2d(32, 1, kernel_size=3, padding=1, stride=1)

        def forward(self, x):
            # passing the image through encoder
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))

            # passing the encoded part through decoder
            x = F.relu(self.conv2d_1(x))
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = F.relu(self.conv2d_2(x))
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = F.sigmoid(self.conv2d_3(x))

            return x


    model = Denoiser()
    model
    # %%
    # defining the loss function
    criterion = nn.MSELoss()

    # defining the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # %%
    if torch.cuda.is_available():
        print('Cuda Available..Training on GPU')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        model = model.cuda()
    else:
        print('CUDA not available..Traning on CPU')
    # %%
    epochs = 10

    training_loss = 0
    min_valid_loss = np.Inf

    cuda_available = torch.cuda.is_available()
    print(cuda_available)

    save_file = "save/model.pt"

    for e in range(epochs):
        print("Training")
        i = 0

        for images, targets in train_loader:
            print("\rTrain", i, end="")

            if cuda_available:
                images, targets = images.cuda(), targets.cuda()

            outputs = model(images)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            i += 1

        print("Validating")
        i = 0

        with torch.no_grad():
            valid_loss = 0

            for images, targets in valid_loader:
                print("\rValidate", i, end="")

                if cuda_available:
                    images, targets = images.cuda(), targets.cuda()

                outputs = model(images)
                loss = criterion(outputs, targets)

                valid_loss += loss.item()

                i += 1

            if valid_loss < min_valid_loss:
                print('Loss Decreased..({:.3f} -> {:.3f})  Saving Model..'.format(valid_loss, min_valid_loss))
                torch.save(model.state_dict(), save_file)
                min_valid_loss = valid_loss / len(valid_loader)

        print('Epoch: {}/{} -- Training Loss: {:.3f} -- Testing Loss: {:.3f}'.format(e + 1, epochs,
                                                                                     training_loss / len(train_loader), \
                                                                                     valid_loss / len(valid_loader)))

        training_loss = 0
    # %%
    # model = Denoiser()
    # model.load_state_dict(torch.load("save/model.pt" ))
    # model = model.cuda()
    # %%
    '''
    test_pairs = []

    for images, targets in test_loader:
        if cuda_available:
            images, targets = images.cuda(), targets.cuda()

        predictions = model(images)

        score = criterion(predictions, targets)

        test_pairs.append((images, targets, predictions.cpu(), score.cpu()))
    '''
    # %%
    """import gc

    # model.cpu()
    del score #, ,  # , , 
    gc.collect()
    torch.cuda.empty_cache()"""