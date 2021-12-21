import random
import time
import copy
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from torchvision import datasets
from torchvision.transforms import transforms


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)


def train_model(model=None, dataloaders=None, real_dataloader=None, criterion=None, optimizer=None, scheduler=None,
                num_epochs=25, label_list=None, real_labels=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss, train_acc, valid_loss, valid_acc, real_acc = [], [], [], [], []

    since = time.time()

    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects, num_cnt = 0.0, 0, 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_cnt += len(labels)
            if phase == 'train':
                scheduler.step()

            epoch_loss = float(running_loss / num_cnt)
            epoch_acc = float((running_corrects.double() / num_cnt).cpu() * 100)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
            print('{} Loss: {:.2f} Acc: {:.2f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('==> best model saved - %d / %.2f' % (best_idx + 1, best_acc))

        model.eval()
        batch_correct, cnt = 0, 0
        for real_inputs, real_label in real_dataloader:
            real_inputs = real_inputs.to(device)
            real_label = real_label.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(False):
                outputs = model(real_inputs)
                _, preds = torch.max(outputs, 1)

            for idx in range(len(preds.data.cpu().numpy())):
                if real_labels[real_label.data.cpu().numpy()[idx]] == label_list[preds[idx]]:
                    batch_correct += 1

            cnt += len(real_label)

        epoch_acc = float((batch_correct / cnt) * 100)

        real_acc.append(epoch_acc)

        print('Real Image Test: {} / {}, Acc - {}'.format(batch_correct, cnt, epoch_acc))

    time_elapsed = time.time() - since
    print('=' * 10)
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: %d - %.1f' % (best_idx + 1, best_acc))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'texture_model.pt')
    print('model saved')

    return model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc, real_acc


class LoadDatset:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.data_path = './data/combine'
        self.texture_dataset = datasets.ImageFolder(self.data_path, transforms.Compose(
            [transforms.Grayscale(3), transforms.Resize((64, 64)), transforms.ToTensor()]))
        self.label_list = self.texture_dataset.classes
        train_idx, valid_idx = train_test_split(list(range(len(self.texture_dataset))), test_size=0.2,
                                                random_state=9608)

        self.dataset = {}
        self.dataset['train'] = Subset(self.texture_dataset, train_idx)
        self.dataset['valid'] = Subset(self.texture_dataset, valid_idx)

    def data_load(self):
        dataloaders = {}
        dataloaders['train'] = DataLoader(self.dataset['train'], batch_size=self.batch_size, shuffle=True,
                                          num_workers=0)
        dataloaders['valid'] = torch.utils.data.DataLoader(self.dataset['valid'], batch_size=self.batch_size,
                                                           shuffle=False, num_workers=0)
        return dataloaders


class RealDataset:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.real_dataset = datasets.ImageFolder('./data/real_images', transforms.Compose(
            [transforms.Grayscale(3), transforms.Resize((64, 64)), transforms.ToTensor()]))
        self.real_labels = self.real_dataset.classes

    def real_data_load(self):
        return torch.utils.data.DataLoader(self.real_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)


def draw_graph(best_idx, train_acc, train_loss, valid_acc, valid_loss, real_acc):
    print('best model : %d - %1.f / %.1f' % (best_idx, valid_acc[best_idx], valid_loss[best_idx]))
    fig = plt.figure()
    ax1 = fig.subplots()

    ax1.plot(train_acc, 'b-', label='train_acc')
    ax1.plot(valid_acc, 'r-', label='valid_acc')
    ax1.plot(real_acc, 'y-', label='real_acc')
    plt.plot(best_idx, valid_acc[best_idx], 'ro')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('acc', color='k')
    ax1.tick_params('y', colors='k')

    ax2 = ax1.twinx()
    ax2.plot(train_loss, 'g-', label='train_loss')
    ax2.plot(valid_loss, 'k-', label='valid_loss')
    plt.plot(best_idx, valid_loss[best_idx], 'ro')
    plt.ylim(-0.1, 10)
    ax2.set_ylabel('loss', color='k')
    ax2.tick_params('y', colors='k')

    fig.legend()

    fig.tight_layout()
    plt.show()


def test_realdata(model=None, real_dataloader=None, label_list=None, real_labels=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    correct, cnt = 0, 0
    for img, label in real_dataloader:
        img = img.to(device)
        label = label.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(img)
            _, preds = torch.max(outputs, 1)

        for idx in range(len(preds.data.cpu().numpy())):
            if real_labels[label[idx].cpu()] == label_list[preds[idx].cpu()]:
                signal = 'O'
            else:
                signal = 'X'
            print("{} Real: {}, Predict: {} - {:.2f}%".format(signal, real_labels[label[idx].cpu()],
                                                              label_list[preds[idx].cpu()],
                                                              torch.max(outputs[idx])))
            if real_labels[label[idx]] == label_list[preds[idx]]:
                correct += 1

        cnt += len(label)
    acc = correct / cnt * 100
    print(' Correct {} / Total {},  Acc: {}'.format(correct, cnt, acc))
