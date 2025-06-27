from tqdm import tqdm
import torch
from sklearn.metrics import log_loss
import torch.nn.functional as F


def evaluation(model, dataloader, val_dataset, criterion, device, epoch, num_epochs, class_names):
    model.eval()  # 모델을 평가 모드로 설정
    valid_loss = 0.0
    valid_accuracy = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad(): # model의 업데이트 막기
        tbar = tqdm(dataloader)
        for images, labels in tbar:
            images = images.to(device)
            labels = labels.to(device)

            # 순전파
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 손실과 정확도 계산
            valid_loss += loss.item()
            # torch.max에서 dim 인자에 값을 추가할 경우, 해당 dimension에서 최댓값과 최댓값에 해당하는 인덱스를 반환
            _, predicted = torch.max(outputs, 1)
            valid_accuracy += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # LogLoss
            probs = F.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # tqdm의 진행바에 표시될 설명 텍스트를 설정
            tbar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Valid Loss: {loss.item():.4f}")

    valid_loss = valid_loss / len(dataloader)
    valid_accuracy = valid_accuracy / len(val_dataset)
    val_logloss = log_loss(all_labels, all_probs, labels=list(range(len(class_names))))

    return model, valid_loss, valid_accuracy, val_logloss