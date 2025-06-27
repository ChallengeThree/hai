import torch
import numpy as np


class EarlyStopping:
    def __init__(self, patience=5, verbose=True, delta=0.0, save_path='checkpoint.pt', enabled=True):
        """
        Args:
            patience (int): 개선이 없을 때 기다릴 에폭 수
            verbose (bool): 개선 시 출력 여부
            delta (float): 개선이라고 간주할 최소 변화량
            save_path (str): 모델을 저장할 경로
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.save_path = save_path
        self.enabled = enabled

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = float('inf')


    def __call__(self, val_loss, model):
        if not self.enabled:
            return  # Early stopping 안 씀
        score = -val_loss  # 손실값이 낮을수록 좋은 경우

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"  ↪️ No improvement. EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
            self.counter = 0

    def _save_checkpoint(self, val_loss, model):
        '''모델 저장'''
        if self.verbose:
            print(f"  ✅ Validation loss improved. Saving model...")
        torch.save(model.state_dict(), self.save_path)
        self.best_loss = val_loss