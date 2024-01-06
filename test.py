## import libraries for training
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
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns


# Validating the model
def evaluate(val_loader,model):
    model.cpu()
    #model.cuda()
    model.eval()
    model.training=False
    map = AverageMeter()
    imageloop = []
    imagelabel = []
    with torch.no_grad():
        for i, (images,target,fnames) in enumerate(val_loader):
            #img = images.cuda(non_blocking=True)
            #label = target.cuda(non_blocking=True)
            img = images.cpu()
            label = target.cpu()
            imageloop.append(img)
            imagelabel.append(label)
            with torch.cuda.amp.autocast():
                logits = model(img)
                preds = logits.softmax(1)
            
            #valid_map5, valid_acc1, valid_acc5, precision, recall = map_accuracy(preds, label)
            #map.update(valid_map5,img.size(0))
             # figure size
        fig = plt.figure(figsize=(20, 15))
        print('Number of frames: {}'.format(len(imageloop)))
        # index of equidistant frames
        # nth_frames = np.linspace(0, len(video_09) - 1, n).astype(int)
        for i in range(len(imageloop)):
            frame = imageloop[0][0][i]
            fig.add_subplot(1, len(imageloop), i + 1)
            plt.imshow(frame, cmap='gray')
            plt.axis('off')
        plt.show()
    return map.avg

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
        predicted_labels = probs.argmax(dim=1)
        # predicted_labels_np = np.array(predicted_labels.cpu())  # Convert from tensor to NumPy array
        # true_labels_np = np.array(truth.cpu())
        precision = precision_score(truth, predicted_labels, average='weighted')
        recall = recall_score(truth, predicted_labels, average='weighted')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        #roc_curve(predicted_labels, truth)
        confusion_matrixfn(predicted_labels, truth)
        return map5, acc1, acc5, precision, recall

def confusion_matrixfn(pred, label):
    cm = confusion_matrix(label, pred)

    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=np.unique(label), yticklabels=np.unique(label))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def roc_curve(pred, label):
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(label, pred)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

######################## load file and get splits #############################
if __name__ == '__main__': 
    print('reading test file')
    test_files = pd.read_csv("test.csv")
    print('Creating test dataloader')
    test_gen = knifeDataset(test_files,mode="val")
    test_loader = DataLoader(test_gen,batch_size=8,shuffle=False,pin_memory=True,num_workers=8)

    print('loading trained model')
    model = timm.create_model('resnet50', pretrained=True,num_classes=config.n_classes)
    map_location=torch.device('cpu')
    model.load_state_dict(torch.load('Knife-RN50-E19.pt', map_location=map_location))
    #model.load_state_dict(torch.load('Knife-Effb0-E1.pt'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ############################# Training #################################
    print('Evaluating trained model')
    map = evaluate(test_loader,model)
    print("mAP =",map)
    
   
