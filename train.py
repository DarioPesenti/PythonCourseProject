import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
import argparse
import wandb
from model import *
from convexity import measure_convexity
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm

def evaluate_model(dataloader, model, criterion, embedding_classifier):
    model.eval() 
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images, return_embedding=True)
            outputs= embedding_classifier(embeddings)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy

def evaluate_model_get_embeddings(dataloader, model, criterion, embedding_classifier):
    model.eval() 
    total_loss, correct, total = 0, 0, 0
    embedding_list, label_list=[], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images, return_embedding=True)
            embedding_list.append(embeddings)
            outputs= embedding_classifier(embeddings)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            label_list.append(predicted)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy, embedding_list, label_list


def train_model(train_loader, test_loader, model,
                embedding_classifier=nn.Linear(16384, 10).to(device),
                criterion=nn.CrossEntropyLoss(), 
                optimizer= torch.optim.Adam(list(model.parameters()) + list(embedding_classifier.parameters()), lr=0.001), 
                num_epochs=50):
    #embeddings_list, labels_list =[], []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        #embedding_epoch, labels_epoch = [], []
        running_loss, correct_pred, total_labels=0, 0, 0
        for images, labels in (train_loader):
            images,labels= images.to(device), labels.to(device)
            optimizer.zero_grad()
            # outputs=model(images)
            extracted_embeddings = model(images, return_embedding=True)
            outputs = embedding_classifier(extracted_embeddings)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()


            running_loss+=loss.item()
            _, predicted= torch.max(outputs.data, 1)
            total_labels+=labels.size(0)
            correct_pred+=(predicted==labels).sum().item()
            
            #embedding_epoch.append(extracted_embeddings.cpu().detach())
            #labels_epoch.append(predicted.cpu().detach())
        #embeddings_list.append(embedding_epoch)
        #labels_list.append(labels_epoch)


        train_accuracy= (correct_pred/total_labels)*100
        if epoch != num_epochs-1: #test at the end of each epoch
            test_loss, test_accuracy = evaluate_model(test_loader, model, criterion, embedding_classifier)
        else:
            test_loss, test_accuracy, embeddings_list, labels_list = evaluate_model_get_embeddings(test_loader, model, criterion, embedding_classifier)
        avg_train_loss = running_loss / len(train_loader)
        if args.wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_accuracy": train_accuracy,
                "train_loss": avg_train_loss,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy
                
            })
        else:
            print("Train accuracy: ", train_accuracy, " Test accuracy: ", test_accuracy)
    return model, embeddings_list, labels_list



def split_and_dataloader(img_batch, label_batch):
    obj_hue_label = label_batch[:, 2]  # Selecting the third column of the labels array
    # First split: Separate out the test set (1000 samples, 3.33% of the total)
    train_val_img, test_img, train_val_label, test_label = train_test_split(img_batch, obj_hue_label, test_size=1/30, random_state=42)

    # Second split: Separate the remaining data into training and the rest (2000 for validation + 27000 for training)
    temp_train_img, val_img, temp_train_label, val_label = train_test_split(train_val_img, train_val_label, test_size=2000/29000, random_state=42)

    # Third split: Now separate out the training set (3000 samples from the remaining 27000) 
    train_img, _, train_label, _ = train_test_split(temp_train_img, temp_train_label, test_size=24000/27000, random_state=42)

    num_classes = 10
    # Convert numpy arrays to PyTorch tensors
    train_img_tensor = torch.tensor(train_img).permute(0, 3, 1, 2)  # Adjust dimensions to [N, C, H, W]
    val_img_tensor = torch.tensor(val_img).permute(0, 3, 1, 2)
    test_img_tensor = torch.tensor(test_img).permute(0, 3, 1, 2)

    train_label_tensor = torch.tensor((train_label * num_classes).astype(int), dtype=torch.long)  # Ensure labels are long type for CrossEntropyLoss
    val_label_tensor = torch.tensor((val_label * num_classes).astype(int), dtype=torch.long)
    test_label_tensor = torch.tensor((test_label * num_classes).astype(int), dtype=torch.long)

    # Create a dataset and dataloader for batch processing
    train_dataset = TensorDataset(train_img_tensor, train_label_tensor)
    val_dataset = TensorDataset(val_img_tensor, val_label_tensor)
    test_dataset = TensorDataset(test_img_tensor, test_label_tensor)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)  # Adjust batch size as needed
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    return train_loader, val_loader, test_loader

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-w','--wandb', action='store_true', help='enable wandb login, default False')
    parser.add_argument('-c','--convexity', action='store_true', help='measure convexity of model embeddings, default False')
    parser.add_argument('-e', '--epochs', type=int, default=20, help='number of training epochs. Int, default 20')
    parser.add_argument('-s', '--seed', default=42, type=int, help='Random seed number. Int, default 42')
    args=parser.parse_args()
    seed_value=args.seed  
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    path= os.getcwd() + '//'
    img_batch = np.load(path + 'img_batch.npy')
    label_batch = joblib.load(path + 'label_batch.pkl')
    train_loader, val_loader, test_loader= split_and_dataloader(img_batch, label_batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #CUDA_VISIBLE_DEVICES=2 # remove if not necessary for your machine, la gente mi intasa i server clic
    if args.wandb:
        wandb.init(project='convexity')

    train_loader, val_loader, test_loader= split_and_dataloader(img_batch, label_batch)
    model, emb, labels = train_model(train_loader, test_loader, model, embedding_classifier, criterion=criterion, optimizer=optimizer, num_epochs=args.epochs)
    if args.convexity:
        measure_convexity(emb, labels, model, device)
