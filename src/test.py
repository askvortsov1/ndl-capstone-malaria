import torch
from tqdm import tqdm


def test_model(model, test_dataloader, optimizer, criterion, device):
    print("Testing Best Model")
    print("=" * 15)
    running_loss = 0
    running_corrects = 0
    model.eval()
    for inputs, labels in tqdm(test_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            # Get model outputs and calculate loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss / len(test_dataloader.dataset)
    test_acc = running_corrects.double() / len(test_dataloader.dataset)

    print('Loss: {:.4f} Acc: {:.4f}'.format(test_loss, test_acc))

    return test_loss, test_acc
