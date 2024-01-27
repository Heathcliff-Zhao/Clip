import clip
import torch
from torch.utils.data import DataLoader
import pandas as pd
# from sklearn.metrics import accuracy_score, recall_score, precision_score
from dataset import ClipDataset, DisturbClipDataset
import wandb

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

def accuracy_score(labels, predictions):
    correct = 0
    for label, prediction in zip(labels, predictions):
        if label == prediction:
            correct += 1
    return correct / len(labels)

def recall_score(labels, predictions):
    # print(labels)
    # print(predictions)
    tp = 0
    fn = 0
    for label, prediction in zip(labels, predictions):
        # print(label, prediction)
        if label == 1 and prediction == 1:
            tp += 1
        elif label == 1 and prediction == 0:
            fn += 1
    return tp / (tp + fn)

def precision_score(labels, predictions):
    tp = 0
    fp = 0
    for label, prediction in zip(labels, predictions):
        if label == 1 and prediction == 1:
            tp += 1
        elif label == 0 and prediction == 1:
            fp += 1
    return tp / (tp + fp)


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
            # print(logits)
            predictions = torch.argmax(logits, dim=-1)
            # print(predictions)
            all_labels.extend(torch.argmax(labels, dim=-1).cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())
            
    # print(all_labels[:50])
    # print(all_predictions[:50])
    # exit()
    print(all_labels)
    print(all_predictions)
    accuracy = accuracy_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    print(f'Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}')
    return accuracy, recall, precision

def train(classifier, train_loader, test_loader, criterion, optimizer, device, text_inputs):
    epochs = 100
    for epoch in range(epochs):
        train_loss = train_epoch(classifier, train_loader, criterion, optimizer, device, text_inputs)
        accuracy, recall, precision = evaluate_performance(classifier, test_loader, device, text_inputs)
        wandb.log({'epoch_loss': train_loss, 'accuracy': accuracy, 'recall': recall, 'precision': precision})
        if (epoch + 1) % 20 == 0:
            torch.save(classifier.state_dict(), f'./models/clip_model_{epoch + 1}.pth')

if __name__ == '__main__':
    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/fixtest.csv')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier, preprocess = clip.load("ViT-B/32", device=device)
    classifier = classifier.float()

    train_dataset = ClipDataset(train_df, preprocess)
    test_dataset = ClipDataset(test_df, preprocess)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    text_inputs = clip.tokenize(['photo of a good webpage', 'photo of a bad webpage']).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=1e-5)
    train(classifier, train_loader, test_loader, criterion, optimizer, device, text_inputs)
