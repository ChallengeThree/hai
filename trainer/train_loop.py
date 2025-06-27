from trainer.training import training
from trainer.evaluation import evaluation
from utils.EarlyStopping import EarlyStopping


def training_loop(model, train_dataloader, valid_dataloader, train_dataset, val_dataset, criterion, optimizer, device, num_epochs, early_stopping: EarlyStopping, class_names):
    valid_max_accuracy = -1

    for epoch in range(num_epochs):
        model, train_loss, train_accuracy = training(model, train_dataloader, train_dataset, criterion, optimizer, device, epoch, num_epochs)
        model, valid_loss, valid_accuracy, val_logloss = evaluation(model, valid_dataloader, val_dataset, criterion, device, epoch, num_epochs, class_names)

        if valid_accuracy > valid_max_accuracy:
          valid_max_accuracy = valid_accuracy
        
        # early stopping 및 check point에서 모델 저장
        early_stopping(val_logloss, model)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}")

        if early_stopping.early_stop:
            print(f"🛑 Early stopping at epoch {epoch + 1}")
            break


    return model, valid_max_accuracy



