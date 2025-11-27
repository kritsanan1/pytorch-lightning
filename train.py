import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from model import LitModel
from data import MNISTDataModule

def main():
    # Initialize data module
    data_module = MNISTDataModule(data_dir='./data', batch_size=64)
    
    # Initialize model
    model = LitModel(input_dim=784, hidden_dim=128, output_dim=10, learning_rate=1e-3)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        filename='best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min'
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=TensorBoardLogger('lightning_logs/', name='mnist_model'),
        accelerator='auto',
        devices='auto'
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    # Test the model
    trainer.test(model, data_module)
    
    print("Training completed!")

if __name__ == '__main__':
    main()