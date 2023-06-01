import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig
import logging
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import time
import torch.nn.functional as F

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path='config', config_name='config_dl')
def my_app(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    params = OmegaConf.to_container(cfg['params'])
    class HeartDataset(Dataset):
        def __init__(self, data, label):
            self.data = data
            self.label = label

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            data = self.data[idx]
            marks = self.label[idx]
            sample = [data, marks]

            return sample
    # Загружаем данные
    X_train = np.load('data/X_train_ohe.npy')
    y_train = np.load('data/Y_train_ohe.npy')
    X_test = np.load('data/X_test_ohe.npy')
    y_test = np.load('data/Y_test_ohe.npy')

    # Создаем Dataloader
    train_dataset = HeartDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_dataset = HeartDataset(X_test, y_test)
    val_dataloader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

    print("len(train_dataset) =", len(train_dataset))
    print("len(val_dataset) =", len(val_dataset))

    drop = params['dropout']
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(3753, 136)
            self.fc2 = nn.Linear(136, 136)
            self.fc3 = nn.Linear(136, 136)
            self.fc4 = nn.Linear(136, 136)
            self.fc5 = nn.Linear(136, 136)
            self.fc6 = nn.Linear(136, 136)
            self.fc7 = nn.Linear(136, 136)
            self.fc8 = nn.Linear(136, 1)
            self.dropout = nn.Dropout(drop)
            #self.fc9 = nn.Linear(102, 1)
            #self.fc10 = nn.Linear(68, 68)
            #self.fc11 = nn.Linear(68, 68)
            #self.fc12 = nn.Linear(68, 68)
            #self.fc13 = nn.Linear(68, 68)
            #self.fc14 = nn.Linear(68, 68)
            #self.fc15 = nn.Linear(68, 68)
            #self.fc16 = nn.Linear(68, 1)
            self.act1 = nn.ReLU()
        def forward(self, x):
            y = self.dropout(self.act1(self.fc1(x)))
            y = self.act1(self.fc2(y))
            y = self.act1(self.fc3(y))
            y = self.act1(self.fc4(y))
            y = self.act1(self.fc5(y))
            y = self.act1(self.fc6(y))
            y = self.act1(self.fc7(y))
            #y = self.act1(self.fc8(y))
            #y = self.act1(self.fc9(y))
            #y = self.act1(self.fc10(y))
            #y = self.act1(self.fc11(y))
            #y = self.act1(self.fc12(y))
            #y = self.act1(self.fc13(y))
            #y = self.act1(self.fc14(y))
            #y = self.act1(self.fc15(y))
            y = F.sigmoid(self.fc8(y))
            return y

    device = torch.device('cuda:0' if torch.cuda else 'cpu')
    print(device)
    #device = 'cpu'
    model = Net()
    model.train()
    model.to(device)
    log.info(model)
    if params['err'] == 'BCE':
        loss_func = nn.BCELoss()
    if params['err'] == 'L1':
        loss_func = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])

    def accuracy(output, labels):
        predictions = torch.round(output)
        correct = (predictions == labels).sum().cpu().numpy()
        return correct / len(labels)

    start_time = time.time()

    for epoch in range(params['epoch']):
        model.train()
        for itr, data in enumerate(train_dataloader):

            imgs = data[0].float().to(device)#.float().to(device)
            labels = data[1].float().to(device)#.float().to(device)
            y_pred = model(imgs)
            labels = labels.reshape(-1,1)
            loss = loss_func(y_pred, labels)

            if itr % 500 == 0:
                log.info(f'Iteration {itr:.0f}, epoch {epoch:.0f}, train accuracy {accuracy(y_pred, labels):.2f}, loss {loss:.4f}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        maxacc = 0
        accuracies = []
        for itr, data in enumerate(val_dataloader):
            imgs = data[0].float().to(device)
            labels = data[1].float().to(device)
            labels = labels.reshape(-1, 1)
            y_pred = model(imgs)
            accuracies.append(accuracy(y_pred, labels))
        log.info(f'Test accuracy - {np.mean(np.array(accuracies))}')
        if np.mean(np.array(accuracies)) > maxacc:
            print('Saving model because its better')
            maxacc = np.mean(np.array(accuracies))
            model_name = params['model']
            checkpoint = {'state_dict': model.state_dict(),'optimizer' :optimizer.state_dict()}
            torch.save(checkpoint, f'{model_name}.pth')
    log.info('Total time {:.4f} seconds'.format(time.time() - start_time))

if __name__ == "__main__":
    my_app()