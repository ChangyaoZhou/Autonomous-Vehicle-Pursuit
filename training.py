import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import random
from torchvision import transforms
import argparse
from torch.utils.tensorboard import SummaryWriter
import json

from data.my_datasets import MyDataset
from prediction.models import MyModel_CNN


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SPLIT_DIR = './train/split/'
LOG_DIR = './train/training_logs/'
CONFIG_PATH = './config.json'



def shuffle_data(delta_file, split_dir):
    train_out = open(split_dir + "traindata_shuffle.txt", 'w')
    val_out = open(split_dir + "valdata_shuffle.txt", 'w')
    over_out = open(split_dir + "overdata_shuffle.txt", 'w')
    lines = []
    with open(delta_file, 'r') as infile:
        for line in infile:
            lines.append(line)
        random.shuffle(lines)
        num_train = np.ceil(0.85*len(lines))
        for count, line in enumerate(lines):
            if count <= num_train:
                if 25 <= count <= 35:
                    over_out.write(line)
                train_out.write(line)
            else:
                val_out.write(line) 
    train_out.close()            
    val_out.close()
    over_out.close()  

    
def run_epoch(model, criterion, optimizer, dataloader, if_train):
    running_loss1, running_loss2, running_loss3 = 0.0, 0.0, 0.0
    for i, data in enumerate(dataloader, 0):
        X, y1, y2, y3 = data
        X = [x.cuda() for x in X]
        y1 = y1.to(device).float()
        y2 = y2.to(device).float()
        y3 = y3.to(device).float()
        if if_train:
            optimizer.zero_grad()
            y_pred = model(X)
            y_pred1, y_pred2, y_pred3 = y_pred[:, 0].float(), y_pred[:, 1].float(), y_pred[:, 2].float()
            loss1 = criterion(y_pred1, y1)
            loss2 = criterion(y_pred2, y2) 
            loss3 = criterion(y_pred3, y3)
            loss = loss1 + loss2 + loss3
            loss.backward()             
            optimizer.step()            
            running_loss1 += loss1.item() 
            running_loss2 += loss2.item() 
            running_loss3 += loss3.item() 
        else:
            y_pred = model(X)
            y_pred1, y_pred2, y_pred3 = y_pred[:, 0].float(), y_pred[:, 1].float(), y_pred[:, 2].float()
            loss1 = criterion(y_pred1, y1)
            loss2 = criterion(y_pred2, y2)
            loss3 = criterion(y_pred3, y3)
            loss = loss1 + loss2 + loss3 
            running_loss1 += loss1.item() 
            running_loss2 += loss2.item() 
            running_loss3 += loss3.item() 
    return running_loss1, running_loss2, running_loss3, loss


def main():
    parser = argparse.ArgumentParser(description='Training settings')
    parser.add_argument('-depth', type=str, help='resource of depth map used for training, choose cdn or gt')
    parser.add_argument('-relative_transform', type=str,
                        help='resource of relative transformation used for training, choose rcnn or gt. ')

    # args in config file
    parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the dataset') 
    parser.add_argument('-batch_size', type=int, default=256, help='batch size')
    parser.add_argument('-image_dir', type=str, help='directory for images')
    parser.add_argument('-label_file', type=str, help='directory for gt delta files')
    parser.add_argument('-save_dir', type=str, help='directory for trained models')
    parser.add_argument('-train', action='store_true', default=True, help='if train') 
    parser.add_argument('-inference', action='store_true', default=True, help='if inference')  
    
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = json.load(f)
        print(config)
    for k, v in config.items():
        args.__setattr__(k, v)
    if not os.path.exists(SPLIT_DIR):
        os.makedirs(SPLIT_DIR)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
     
    print('device:', device)
    # prepare dataset
    if args.shuffle:
        shuffle_data(args.delta_file, SPLIT_DIR)
        
    img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_size = (128, 128)
    over_data = MyDataset(root=args.image_dir, txtname=SPLIT_DIR + 'overdata_shuffle.txt', transform=img_transform, size=img_size)
    train_data = MyDataset(root=args.image_dir, txtname=SPLIT_DIR + 'traindata_shuffle.txt', transform=img_transform, size=img_size)
    val_data = MyDataset(root=args.image_dir, txtname=SPLIT_DIR + 'valdata_shuffle.txt', transform=img_transform, size=img_size)
    over_loader = DataLoader(dataset=over_data, batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=True)
    print("Overfitting dataset size: %i" % len(over_data))
    print("Training dataset size: %i" % len(train_data))
    print("Validation dataset size: %i" % len(val_data))
    num_trainbatch = np.ceil(len(train_data)/args.batch_size) 
    num_valbatch = np.ceil(len(val_data)/args.batch_size) 
    
    # load the model
    mynet = MyModel_CNN().to(device)
    total_trainable_params = sum(
        p.numel() for p in mynet.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} trainable parameters.') 
    
    # training settings    
    train_history = np.empty([0, 3], dtype=float) 
    val_history = np.empty([0, 3], dtype=float)
    mycriterion = nn.MSELoss(reduction='mean') 
    myoptimizer = optim.Adam(mynet.parameters(), lr=1e-3, eps=1e-08)
    
    # tensorboard setup
    writer = SummaryWriter(LOG_DIR + args.depth + '_' + args.relative_transform)
    
    # start training
    if args.train:
        max_epochs = 120
        print('Start training!')
        for epoch in range(max_epochs):            
            if epoch >= 20:                
                myoptimizer.param_groups[0]['lr'] = 1e-4
            if epoch >= 75:               
                myoptimizer.param_groups[0]['lr'] = 1e-6
            if epoch == 20:
                torch.save(mynet.state_dict(), args.save_dir + args.depth + '_' + args.relative_transform + '_20.pth')
                print('Model with 20 epochs is saved!')
            if epoch == 75:
                torch.save(mynet.state_dict(), args.save_dir + args.depth + '_' + args.relative_transform + '_75.pth')
                print('Model with 60 epochs is saved!')
                
            train_loss1, train_loss2, train_loss3, loss = run_epoch(model=mynet, criterion=mycriterion, optimizer=myoptimizer, dataloader=train_loader, iftrain=True)
            # log tensorboard
            writer.add_scalar('train/train_loss_av', loss/num_trainbatch, epoch)
            writer.add_scalar('train/train_loss1', train_loss1/num_trainbatch, epoch)
            writer.add_scalar('train/train_loss2', train_loss2/num_trainbatch, epoch)
            writer.add_scalar('train/train_loss3', train_loss3/num_trainbatch, epoch)

            train_history = np.append(train_history, np.array([[train_loss1, train_loss2, train_loss3]])/num_trainbatch, axis=0)
            val_loss1, val_loss2, val_loss3 = run_epoch(model=mynet, criterion=mycriterion, optimizer=myoptimizer, dataloader=val_loader, iftrain=False)
            val_loss = (val_loss1 + val_loss2 + val_loss3) / 3
            # log tensorboard
            writer.add_scalar('validation/val_loss_av', val_loss/num_valbatch, epoch)
            writer.add_scalar('validation/val_loss1', val_loss1/num_valbatch, epoch)
            writer.add_scalar('validation/val_loss2', val_loss2/num_valbatch, epoch)
            writer.add_scalar('validation/val_loss3', val_loss3/num_valbatch, epoch)
            val_history = np.append(val_history, np.array([[val_loss1, val_loss2, val_loss3]])/num_valbatch, axis=0)
            print(f"Epoch {epoch + 1: >3}/{max_epochs}")
            print('Delta X: train_loss: %2e, val_loss: %2e' % (train_history[-1][0], val_history[-1][0]))
            print('Delta Y: train_loss: %2e, val_loss: %2e' % (train_history[-1][1], val_history[-1][1]))
            print('Delta yaw: train_loss: %2e, val_loss: %2e' % (train_history[-1][2], val_history[-1][2]))
        print('FINISH.')

        # save trained model
        torch.save(mynet.state_dict(), args.save_dir + args.depth + '_' + args.relative_transform + '_120.pth')
        print('Model with 100 epochs is saved!')
    
    if args.inference:
        mynet.load_state_dict(torch.load(args.save_dir + args.depth + '_' + args.relative_transform + '_120.pth'))
        # inference
        for i in range(11):
            test = random.randint(0, len(over_data))
            test_in, gt_x, gt_y, gt_yaw = val_data[test]
            test_in = [torch.tensor(x).cuda() for x in test_in]
            gt_output = [gt_x, gt_y, gt_yaw]
            gt_output = torch.tensor(gt_output).to(device)
            output_pred = mynet(test_in) 
            print('prediction:{}, ground truth:{}'.format(output_pred[:, :3].cpu().detach().numpy(), gt_output[:3].cpu().detach().numpy()))
    

if __name__ == '__main__':
    main()
