import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(get_available_gpu())

import clip
import torch
from torch.utils.data import DataLoader
import pandas as pd
from dataset import ClipDataset, DisturbClipDataset
import wandb
import argparse

from utils import accuracy_score, recall_score, precision_score, get_available_gpu

wandb.init(project='clip-website-classification')


def train_epoch(model, data_loader, criterion, optimizer, device, text_inputs):
    model.train()
    total_loss = 0
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        # images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, text_inputs)
        logits1 = outputs[0].softmax(dim=-1)
        logits2 = outputs[1].t().softmax(dim=-1)
        # print(logits)
        # print(labels)
        loss = (criterion(logits1.squeeze(), labels.float()) + criterion(logits2.squeeze(), labels.float())) / 2
        # print(loss)
        wandb.log({'loss': loss})
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # exit()
    return total_loss / len(data_loader)

def evaluate_performance(model, data_loader, device, text_inputs):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images, text_inputs)
            logits = outputs[0].softmax(dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            all_labels.extend(torch.argmax(labels, dim=-1).cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())
            
    print(all_labels)
    print(all_predictions)
    accuracy = accuracy_score(all_labels, all_predictions)
    bad_recall, good_recall = recall_score(all_labels, all_predictions)
    bad_precision, good_precision = precision_score(all_labels, all_predictions)
    print(f'Accuracy: {accuracy}, Recall@bad: {bad_recall}, Recall@good: {good_recall}, Precision@bad: {bad_precision}, Precision@good: {good_precision}')
    return accuracy, bad_recall, good_recall, bad_precision, good_precision

def train(classifier, train_loader, test_loader, criterion, optimizer, device, text_inputs):
    epochs = 60
    for epoch in range(epochs):
        train_loss = train_epoch(classifier, train_loader, criterion, optimizer, device, text_inputs)
        accuracy, bad_recall, good_recall, bad_precision, good_precision = evaluate_performance(classifier, test_loader, device, text_inputs)
        wandb.log({'train_loss': train_loss, 'accuracy': accuracy, 'bad_recall': bad_recall, 'good_recall': good_recall, 'bad_precision': bad_precision, 'good_precision': good_precision})
        if (epoch + 1) % 20 == 0:
            torch.save(classifier.state_dict(), f'./models/clip_model_{epoch + 1}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_disturb_dataset', action='store_true')
    args = parser.parse_args()

    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/fixtest.csv')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier, preprocess = clip.load("ViT-B/32", device=device)
    classifier = classifier.float()

    train_dataset = ClipDataset(train_df, preprocess, split_train=False) if not args.use_disturb_dataset else DisturbClipDataset(train_df, preprocess)
    test_dataset = ClipDataset(test_df, preprocess)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    text_inputs = clip.tokenize(['photo of a good webpage', 'photo of a bad webpage']).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=1e-5)
    train(classifier, train_loader, test_loader, criterion, optimizer, device, text_inputs)
