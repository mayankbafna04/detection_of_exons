import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from pathlib import Path
import os
import matplotlib.pyplot as plt
from avoa_optimizer import AVOAOptimizer
from data_preperation import process_genbank

def load_data(data_dir):
    """Load processed data with type casting"""
    X = np.load(f"{data_dir}/sequences.npy").astype(np.float32)
    y = np.load(f"{data_dir}/labels.npy").astype(np.float32)
    return X, y

def main():
    # Create directory structure
    Path("processed_data").mkdir(exist_ok=True, parents=True)
    Path("outputs/models").mkdir(exist_ok=True, parents=True)
    Path("outputs/plots").mkdir(exist_ok=True, parents=True)
    
    # Process data if needed
    if not os.path.exists("processed_data/sequences.npy"):
        print("Processing GenBank files...")
        process_genbank("data/genbank_files", "processed_data")
    
    # Load and prepare data
    X, y = load_data("processed_data")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # AVOA optimization
    try:
        print("Starting AVOA optimization...")
        avoa = AVOAOptimizer(pop_size=10, max_iter=20)
        best_config = avoa.optimize(X_train, y_train)
        
        if best_config is None:
            raise RuntimeError("Optimization failed to find valid architecture")
        
        # Train final model
        print("Training final model...")
        final_model = best_config.create_model(X_train.shape[1:])
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)
        ]
        
        history = final_model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=64,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save and evaluate
        final_model.save("outputs/models/best_model.keras")
        
        # Calculate final BER
        y_pred = (final_model.predict(X_test) > 0.5).astype(int).flatten()
        fn = np.sum((y_test == 1) & (y_pred == 0))
        fp = np.sum((y_test == 0) & (y_pred == 1))
        ber = (fp + fn) / len(y_test)
        print(f"Final Test BER: {ber:.4f}")
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('outputs/plots/training_history.png')
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
