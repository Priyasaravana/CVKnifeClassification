## import libraries for training
import sys
import warnings
from datetime import datetime
from timeit import default_timer as timer
import pandas as pd
import torch.optim
from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data import knifeDataset
import timm
from utils import *
warnings.filterwarnings('ignore')
from args import argument_parser
parser = argument_parser()
args = parser.parse_args()

## Writing the loss and results
if not os.path.exists("./logs/"):
    os.mkdir("./logs/")
log = Logger()
filename = "log_train" + str(args.epochs) + ".txt"
log.open(os.path.join("logs", filename))
log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write('                           |----- Train -----|----- Valid----|----- Valid----|---------|\n')
log.write('mode     iter     epoch    |       loss      |        mAP    |      Acc      | time    |\n')
log.write('-------------------------------------------------------------------------------------------------------------------------\n')

## Training the model
def train(train_loader,model,criterion,optimizer,epoch,valid_accuracy,start):
    losses = AverageMeter()
    model.train()
    model.training=True
    for i,(images,target,fnames) in enumerate(train_loader):
        img = images.cpu()
        label = target.cpu()
        
        #with torch.cuda.amp.autocast():
        logits = model(img)
        loss = criterion(logits, label)
        losses.update(loss.item(),images.size(0))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()        
        optimizer.zero_grad()
        scheduler.step()

        print('\r',end='',flush=True)
        message = '%s %5.1f %6.1f        |      %0.3f     |      %0.3f     | %s' % (\
                "train", i, epoch,losses.avg,valid_accuracy[0],valid_accuracy[1], time_to_str((timer() - start),'min'))
        print(message , end='',flush=True)
    log.write("\n")
    log.write(message)

    return [losses.avg]

# Validating the model
def evaluate(val_loader,model,criterion,epoch,train_loss,start):
    model.cuda()
    model.eval()
    model.training=False
    map = AverageMeter()
    
    with torch.no_grad():
        for i, (images,target,fnames) in enumerate(val_loader):
            img = images.cuda(non_blocking=True)
            label = target.cuda(non_blocking=True)
            
            with torch.cuda.amp.autocast():
                logits = model(img)
                preds = logits.softmax(1)
            
            valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, label)
            map.update(valid_map5,valid_acc1,img.size(0))
            
            print('\r',end='',flush=True)
            message = '%s   %5.1f %6.1f       |      %0.3f     |      %0.3f    |      %0.3f    | %s' % (\
                    "val", i, epoch, train_loss[0], map.avg,valid_acc1,time_to_str((timer() - start),'min'))
            print(message, end='',flush=True)
        log.write("\n")  
        log.write(message)
    return [map.avg]

## Computing the mean average precision, accuracy 
def map_accuracy(probs, truth, k=5):
    with torch.no_grad():
        value, top = probs.topk(k, dim=1, largest=True, sorted=True)
        correct = top.eq(truth.view(-1, 1).expand_as(top))

        # top accuracy
        correct = correct.float().sum(0, keepdim=False)
        correct = correct / len(truth)

        accs = [correct[0], correct[0] + correct[1] + correct[2] + correct[3] + correct[4]]
        map5 = correct[0] / 1 + correct[1] / 2 + correct[2] / 3 + correct[3] / 4 + correct[4] / 5
        acc1 = accs[0]
        acc5 = accs[1]
        return map5, acc1, acc5


def my_collate(batch): # batch size 4 [{tensor image, tensor label},{},{},{}] could return something like G = [None, {},{},{}]
    batch = list(filter (lambda x:x is not None, batch)) # this gets rid of nones in batch. For example above it would result to G = [{},{},{}]
    # I want len(G) = 4
    # so how to sample another dataset entry?
    return torch.utils.data.dataloader.default_collate(batch) 

######################## load file and get splits #############################
if __name__ == '__main__': 
    train_imlist = pd.read_csv("train.csv")
    train_gen = knifeDataset(train_imlist,mode="train")
    train_loader = DataLoader(train_gen,batch_size=args.batch_size,shuffle=True,pin_memory=True,num_workers=8)
    #train_loader = DataLoader(train_gen,batch_size=config.batch_size,shuffle=True,pin_memory=True,num_workers=8, collate_fn=my_collate)
    val_imlist = pd.read_csv("test.csv")
    val_gen = knifeDataset(val_imlist,mode="val")
    val_loader = DataLoader(val_gen,batch_size=args.batch_size,shuffle=False,pin_memory=True,num_workers=8)

    ## Loading the model to run
    model = timm.create_model('tf_efficientnet_b0', pretrained=True,num_classes=args.n_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ############################# Parameters #################################
    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs * len(train_loader), eta_min=0,last_epoch=-1)
    criterion = nn.CrossEntropyLoss().cuda()

    ############################# Training #################################
    start_epoch = 0
    val_metrics = [0]
    scaler = torch.cuda.amp.GradScaler()
    start = timer()
    #train
    for epoch in range(0,args.epochs):
        lr = get_learning_rate(optimizer)
        train_metrics = train(train_loader,model,criterion,optimizer,epoch,val_metrics,start)
        val_metrics = evaluate(val_loader,model,criterion,epoch,train_metrics,start)
        ## Saving the model
        filename = "Knife-Effb0-E" + str(epoch + 1)+  ".pt"
        #torch.save(model.state_dict(), filename)
    

   
