class EarlyStopping:
    
    def __init__(self, patience, train_val_diff, val_improvement_threshold, warm_up_epochs, skip_train_val_overfit_check):
        self.patience = patience
        self.train_val_diff = train_val_diff
        self.val_improvement_threshold = val_improvement_threshold
        self.warm_up_epochs = warm_up_epochs
        self.skip_train_val_overfit_check = skip_train_val_overfit_check
        self.counter = 0
        self.best_val_loss = float('inf')
        
    def check_performance(self, epoch, train_loss, val_loss):

        # Check overfitting: val loss diverging from train loss
        if not self.skip_train_val_overfit_check:
            overfitting = (val_loss - train_loss) > self.train_val_diff
        else:
            overfitting = False
        
        # Check plateau: val loss not improving enough
        val_improvement = self.best_val_loss - val_loss
        not_improving = val_improvement < self.val_improvement_threshold
        
        if overfitting or not_improving:
            if epoch > self.warm_up_epochs:
                self.counter += 1
                print(f"No improvement ({self.counter}/{self.patience})")
            else:
                print(f"No improvement (warm-up epoch - {epoch}/{self.warm_up_epochs})")
        else:
            self.best_val_loss = val_loss
            self.counter = 0  # reset if improvement found
            
        if self.counter >= self.patience:
            print(f"\nEarly stopping triggered after {self.patience} epochs without improvement.")
            return True, False

        if overfitting or not_improving:
            return False, False
            
        return False, True