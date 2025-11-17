# Code adapted from the HABITAT project (https://github.com/geoffxy/habitat)

import torch
from torch.utils.data import random_split, DataLoader
import numpy as np
from pathlib import Path
import datetime
from torch.utils.tensorboard import SummaryWriter
import json
from ..Dataset.dataset import Dataset

class Trainer:
    def __init__(self, model, save_path, log_dir):
        self.model = model
        self.device = model.device
        # create save dir
        self.save_dir = Path(save_path)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        with open(self.save_dir/f"config.json", "w") as outfile: 
            json.dump(self.model.config, outfile, indent=4)
        self.log_dir = Path(log_dir)

    def train(self, trainset_path, epochs, testset_path_list=[]):

        # convert to dataset
        dataset = Dataset(trainset_path)
        # print(dataset.df.columns)
        dataset.set_features(self.model.features)
        # print("dataset loaded")

        # train/val split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train, val = random_split(dataset, (train_size, val_size))

        self.train_dataloader = DataLoader(train, batch_size=self.model.train_batch, shuffle=True)
        self.val_dataloader = DataLoader(val, batch_size=self.model.val_batch, shuffle=False)

        # implement losses and optimizers
        def MAPELoss(pred, target):
            return torch.mean(torch.abs((target - pred) / target))
        def SMAPELoss(pred, target):
            return torch.mean(torch.abs((target - pred) / (target + pred)))
        def MSELoss(pred, target, modifier=1):
            se = (target - pred) ** 2
            return torch.mean(se * modifier)
        def RMSELoss(pred, target):
            return torch.sqrt(torch.mean((target - pred) ** 2))
        def MAELoss(pred, target):
            return torch.mean(torch.abs(target - pred))
        def LOGACCLoss(pred, target):
            return torch.mean(torch.square(torch.log(pred/target)))
        loss_dict = {
            "MAPE" : MAPELoss,
            "SMAPE" : SMAPELoss,
            "MSE" : MSELoss,
            "RRMSE" : RMSELoss,
            "MAE" : MAELoss,
            "LOG" : LOGACCLoss,
        }
        lr = self.model.lr
        self.criterion = loss_dict[self.model.loss]
        
        optim = torch.optim.AdamW

        self.optim = optim(self.model.parameters(), lr=lr)

        # Get the current time
        now = datetime.datetime.now()
        formatted_time = now.strftime('%m:%d:%H:%M')

        # set up tensorboard logging
        self.writer = SummaryWriter(self.log_dir/f"{self.model.name}_{formatted_time}")

        # run training loops
        min_perc_err = 1e9
        for epoch in range(epochs):

            # train
            self.model = self.model.train()
            losses = []

            for batch_o, batch_x, batch_m, batch_y in self.train_dataloader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                self.optim.zero_grad()

                pred = self.model(batch_o, batch_x, batch_m.to(self.device))

                # Fix: handle case where model returns a list instead of a tensor
                print(type(pred))
                if isinstance(pred, list):
                    pred = pred[0]

                # Ensure pred is on the correct device
                pred = pred.to(self.device)

                loss = self.criterion(pred, batch_y.reshape(pred.shape))
                loss.backward()
                losses.append(loss.item())
                self.optim.step()


            """
            for batch_o, batch_x, batch_m, batch_y in self.train_dataloader:
                batch_x = batch_x.float()
                batch_y = batch_y.float()

                self.optim.zero_grad()
                pred = self.model(batch_o, batch_x.to(self.device), batch_m.to(self.device))
                loss = self.criterion(pred, batch_y.reshape(pred.shape).to(self.device))
                loss.backward()
                losses.append(loss.item())
                self.optim.step()
            """

            train_loss = sum(losses) / len(losses)

            stats = self.model.stats()

            print("epoch %3s\tloss: %.4f" % (str(epoch), train_loss), end="\t")
            self.writer.add_scalar("train_loss", train_loss, epoch)

            for k, v in stats.items():
                self.writer.add_scalar(k, v, epoch)

            # validate
            self.model = self.model.eval()
            perc_errors = []
            for batch_o, batch_x, batch_m, batch_y in self.val_dataloader:
                batch_x = batch_x.float()
                batch_y = batch_y.float().numpy()

                #pred = self.model(batch_o, batch_x.to(self.device), batch_m.to(self.device)).detach().cpu().numpy()
                pred = self.model(batch_o, batch_x.to(self.device), batch_m.to(self.device))

                # handle list outputs
                if isinstance(pred, list):
                    pred = pred[0]

                pred = pred.detach().cpu().numpy()

                pred = pred.reshape(batch_y.shape)
                pred = np.maximum(pred, 0) # for habitat

                perc_error = np.divide(np.abs(pred - batch_y), batch_y)
                perc_errors.append(perc_error)

            perc_errors_np = np.concatenate(perc_errors)
            avg_err = float(np.mean(perc_errors_np))
            max_err = np.amax(perc_errors_np)

            print("val avg: %.4f, max: %.4f" % (avg_err, max_err), end="\t")
            self.writer.add_scalar("validation avg_err", avg_err, epoch)
            self.writer.add_scalar("validation max_err", max_err, epoch)

            # save model if good
            # save model based on num block test acc
            if avg_err < min_perc_err:
                print("\t(new best, saving!)")
                min_perc_err = avg_err
                self.model.save_state(self.save_dir/"model.pth")
            else:
                continue

        self.writer.close()

    def test(self, testset_path_list=[], out_path=None):

        self.test_dataloaders = []
        for testset_path in testset_path_list:
            testset = Dataset(testset_path)
            testset.set_features(self.model.features)
            dataloader = DataLoader(testset, batch_size=self.model.val_batch, shuffle=False)
            self.test_dataloaders.append((dataloader, testset_path))
        print("dataset loaded")

        # implement losses and optimizers
        def MAPELoss(pred, target):
            return torch.mean(torch.abs((target - pred) / target))
        
        # run training loops
        # test
        self.model = self.model.eval()
        entries = {}
        preds = []
        for test_dataloader, name in self.test_dataloaders:
            perc_errors = []
            for batch_o, batch_x, batch_m, batch_y in test_dataloader:
                batch_x = batch_x.float()
                batch_y = batch_y.float().numpy()

                pred = self.model(batch_o, batch_x.to(self.device), batch_m.to(self.device)).detach().cpu().numpy()
                pred = pred.reshape(batch_y.shape)
                pred = np.maximum(pred, 0) # for habitat
                preds = np.concatenate([preds, pred])

                perc_error = np.divide(np.abs(pred - batch_y), batch_y)
                perc_errors.append(perc_error)

            perc_errors_np = np.concatenate(perc_errors)
            mean_perc_err = float(np.mean(perc_errors_np))
            max_perc_err = np.amax(perc_errors_np)
            print(f"{name} avg: %.4f, max: %.4f" % (mean_perc_err, max_perc_err), end="\t")

            entries[name+"_avg"] = mean_perc_err
            entries[name+"_max"] = max_perc_err
        
        print(preds)

