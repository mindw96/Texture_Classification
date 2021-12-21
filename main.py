import torch
from utils import seed_everything, LoadDatset, RealDataset, train_model, draw_graph, test_realdata
from models import MyModel

seed_everything(9608)

batch_size = 256

dataset = LoadDatset(batch_size)
data_loaders = dataset.data_load()
label_list = dataset.label_list

real_dataset = RealDataset(batch_size)
real_data_loader = real_dataset.real_data_load()
real_labels = real_dataset.real_labels

model = MyModel()

criterion = torch.nn.CrossEntropyLoss()

optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.000001, weight_decay=1e-4)

lr_lambda = lambda epoch: 0.98739
exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_adam, lr_lambda=lr_lambda)

model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc, real_acc = train_model(model=model,
                                                                                                dataloaders=data_loaders,
                                                                                                real_dataloader=real_data_loader,
                                                                                                criterion=criterion,
                                                                                                optimizer=optimizer_adam,
                                                                                                scheduler=exp_lr_scheduler,
                                                                                                num_epochs=500,
                                                                                                label_list=label_list,
                                                                                                real_labels=real_labels)

draw_graph(best_idx, train_acc, train_loss, valid_acc, valid_loss, real_acc)
test_realdata(model=model, real_dataloader=real_data_loader, label_list=label_list, real_labels=real_labels)
