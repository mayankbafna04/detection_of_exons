import os
import numpy as np
import tensorflow as tf
from preprocess import preprocess_data
from avoa import AVOA, optimize_hyperparameters, save_model, load_and_evaluate_model
import matplotlib.pyplot as plt

def main():
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Preprocess data
    print("Step 1: Preprocessing data...")
    genbank_file = "genbank_files"  
    
    # Check if preprocessed data exists
    if (os.path.exists('data/X_train.npy') and 
        os.path.exists('data/X_test.npy') and 
        os.path.exists('data/y_train.npy') and 
        os.path.exists('data/y_test.npy')):
        
        print("Loading preprocessed data...")
        X_train = np.load('data/X_train.npy')
        X_test = np.load('data/X_test.npy')
        y_train = np.load('data/y_train.npy')
        y_test = np.load('data/y_test.npy')
    else:
        print("Preprocessing from scratch...")
        X_train, X_test, y_train, y_test = preprocess_data(genbank_file)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Initialize AVOA
    print("\nStep 2: Initializing AVOA...")
    # For testing, use small population and iterations
    # For real optimization, increase these values
    avoa = AVOA(population_size=5, max_iter=10)
    
    # Run AVOA optimization for CNN structure
    print("\nStep 3: Running AVOA optimization for CNN structure...")
    best_structure, best_fitness = avoa.optimize(X_train, y_train, X_test, y_test)
    
    print(f"\nBest structure fitness: {best_fitness}")
    print("Best structure:")
    print(f"- Convolutional layers: {len(best_structure['conv_layers'])}")
    print(f"- Pooling layers: {len(best_structure['pool_layers'])}")
    print(f"- Fully connected layers: {len(best_structure['fc_layers'])}")
    
    # Optimize hyperparameters
    print("\nStep 4: Optimizing hyperparameters...")
    optimized_structure = optimize_hyperparameters(best_structure, X_train, y_train, X_test, y_test)
    
    # Build and train final model
    print("\nStep 5: Training final model...")
    final_model = avoa.build_cnn_model(optimized_structure)
    
    # Train final model with more epochs
    history = final_model.fit(X_train, y_train,
                             validation_data=(X_test, y_test),
                             epochs=20,
                             batch_size=32,
                             verbose=1)
    
    # Save model
    save_model(final_model)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'])
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'])
    
    plt.tight_layout()
    plt.savefig('results/final_model_training.png')
    
    # Evaluate final model
    print("\nStep 6: Evaluating final model...")
    accuracy, precision, recall, f1_score = load_and_evaluate_model('models/best_model.h5', X_test, y_test)
    
    # Save metrics to file
    with open('results/final_metrics.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1_score:.4f}\n")
    
    print("\nExon detection pipeline completed!")

if __name__ == "__main__":
    main()