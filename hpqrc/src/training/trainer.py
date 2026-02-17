"""
Unified Trainer for RC and DL Models

Supports both:
- RC-style: Extract features → Fit readout (no backprop)
- DL-style: Standard PyTorch training loop
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List
from tqdm import tqdm
import numpy as np


class Trainer:
    """Unified trainer for RC and DL models."""
    
    def __init__(
        self,
        model: Any,
        config: Dict,
        device: str = "cuda",
        callbacks: Optional[List] = None,
    ):
        """
        Args:
            model: Model to train (RC or DL style)
            config: Training configuration
            device: Device to train on
            callbacks: List of callbacks
        """
        self.model = model
        self.config = config
        self.device = device
        self.callbacks = callbacks or []
        
        # Detect model type
        self.is_rc_model = self._detect_rc_model()
        
        # Setup
        if not self.is_rc_model:
            self._setup_dl_training()
        
        # Metrics history
        self.history = {
            "train_loss": [],
            "val_loss": [],
        }
    
    def _detect_rc_model(self) -> bool:
        """Detect if model is RC-style (no backprop)."""
        # Check for RC-specific methods
        if hasattr(self.model, 'extract_features'):
            return True
        if hasattr(self.model, 'readout') and hasattr(self.model, 'fit'):
            return True
        return False
    
    def _setup_dl_training(self):
        """Setup for PyTorch training."""
        self.model = self.model.to(self.device)
        
        # Optimizer
        optimizer_name = self.config.get("optimizer", "adam")
        lr = self.config.get("lr", 0.001)
        
        if optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=self.config.get("momentum", 0.9)
            )
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Loss
        loss_name = self.config.get("loss", "mse")
        if loss_name == "mse":
            self.criterion = nn.MSELoss()
        elif loss_name == "mae":
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.MSELoss()
        
        # Scheduler
        if self.config.get("use_scheduler", False):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=5,
                factor=0.5,
            )
        else:
            self.scheduler = None
        
        # Early stopping
        self.patience = self.config.get("patience", 10)
        self.best_val_loss = float('inf')
        self.counter = 0
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        
        Returns:
            Training history
        """
        if self.is_rc_model:
            return self._fit_rc(train_loader, val_loader)
        else:
            return self._fit_dl(train_loader, val_loader)
    
    def _fit_rc(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
    ) -> Dict:
        """Train RC-style model (feature extraction + readout fitting)."""
        # Collect all training data
        X_train, y_train = self._collect_data(train_loader)
        
        # Fit the model (includes feature extraction + readout)
        print("Fitting RC model...")
        self.model.fit(X_train, y_train)
        
        # Validate if provided
        if val_loader is not None:
            X_val, y_val = self._collect_data(val_loader)
            y_pred = self.model.predict(X_val)
            
            val_loss = np.mean((y_val - y_pred) ** 2)
            self.history["val_loss"].append(val_loss)
            print(f"Validation MSE: {val_loss:.6f}")
        
        return self.history
    
    def _fit_dl(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
    ) -> Dict:
        """Train DL-style model (backpropagation)."""
        n_epochs = self.config.get("epochs", 100)
        
        for epoch in range(n_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
                X, y = batch
                X = X.to(self.device)
                y = y.to(self.device)
                
                self.optimizer.zero_grad()
                pred = self.model(X)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * X.size(0)
            
            train_loss /= len(train_loader.dataset)
            self.history["train_loss"].append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self._evaluate(val_loader)
                self.history["val_loss"].append(val_loss)
                
                # Learning rate scheduling
                if self.scheduler:
                    self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.counter = 0
                else:
                    self.counter += 1
                
                if self.counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}")
        
        return self.history
    
    def _collect_data(self, loader: DataLoader) -> tuple:
        """Collect all data from loader."""
        X_all, y_all = [], []
        
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                X, y = batch
            else:
                X = batch
                y = None
            
            X_all.append(X.numpy() if isinstance(X, torch.Tensor) else X)
            if y is not None:
                y_all.append(y.numpy() if isinstance(y, torch.Tensor) else y)
        
        X = np.vstack(X_all)
        y = np.vstack(y_all) if y_all else None
        
        return X, y
    
    def _evaluate(self, loader: DataLoader) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in loader:
                X, y = batch
                X = X.to(self.device)
                y = y.to(self.device)
                
                pred = self.model(X)
                loss = self.criterion(pred, y)
                
                val_loss += loss.item() * X.size(0)
        
        return val_loss / len(loader.dataset)
    
    def predict(self, loader: DataLoader) -> np.ndarray:
        """Make predictions."""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                X = batch[0] if isinstance(batch, (list, tuple)) else batch
                X = X.to(self.device)
                
                if self.is_rc_model:
                    pred = self.model.predict(X.numpy())
                else:
                    pred = self.model(X).cpu().numpy()
                
                predictions.append(pred)
        
        return np.vstack(predictions)
