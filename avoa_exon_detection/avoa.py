import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

class AVOA:
    def __init__(self, population_size=10, max_iter=50, 
                 minc=2, maxc=5, minp=1, maxp=3, minf=1, maxf=3):
        # AVOA parameters
        self.population_size = population_size
        self.max_iter = max_iter
        
        # CNN structure bounds
        self.minc = minc  # Min number of convolutional layers
        self.maxc = maxc  # Max number of convolutional layers
        self.minp = minp  # Min number of pooling layers
        self.maxp = maxp  # Max number of pooling layers
        self.minf = minf  # Min number of fully connected layers
        self.maxf = maxf  # Max number of fully connected layers
        
        # Results tracking
        self.fitness_history = []
        self.best_structures = []
        
    def initialize_population(self):
        """Initialize random numerical population"""
        population = np.zeros((self.population_size, 3))
        
        for i in range(self.population_size):
            # L1: Number of convolutional layers
            population[i, 0] = np.random.randint(self.minc, self.maxc + 1)
            # L2: Number of pooling layers
            population[i, 1] = np.random.randint(self.minp, self.maxp + 1)
            # L3: Number of fully connected layers
            population[i, 2] = np.random.randint(self.minf, self.maxf + 1)
        
        return population
    
    def create_cnn_structure(self, particle):
        """Convert numerical particle to CNN structure"""
        num_conv = int(particle[0])
        num_pool = int(particle[1])
        num_fc = int(particle[2])
        
        # Define structure parameters
        structure = {
            'conv_layers': [],
            'pool_layers': [],
            'fc_layers': []
        }
        
        # Generate convolutional layers
        for _ in range(num_conv):
            filters = np.random.choice([16, 32, 64, 128])
            kernel_size = np.random.choice([3, 5, 7])
            structure['conv_layers'].append({
                'filters': filters,
                'kernel_size': kernel_size,
                'activation': 'relu'
            })
        
        # Generate pooling layers
        for _ in range(num_pool):
            pool_size = np.random.choice([2, 3])
            structure['pool_layers'].append({
                'pool_size': pool_size
            })
        
        # Generate fully connected layers
        for _ in range(num_fc):
            units = np.random.choice([32, 64, 128, 256])
            dropout = np.random.uniform(0, 0.5)
            structure['fc_layers'].append({
                'units': units,
                'activation': 'relu',
                'dropout': dropout
            })
        
        return structure
    
    def build_cnn_model(self, structure, input_shape=(16, 16, 1)):
        """Build CNN model based on structure"""
        model = Sequential()
        
        # Add first convolutional layer
        if len(structure['conv_layers']) > 0:
            conv_params = structure['conv_layers'][0]
            model.add(Conv2D(filters=conv_params['filters'],
                            kernel_size=conv_params['kernel_size'],
                            activation=conv_params['activation'],
                            input_shape=input_shape))
        
        # Add remaining layers (interleave conv and pool)
        pool_idx = 0
        for i in range(1, len(structure['conv_layers'])):
            # Add convolutional layer
            conv_params = structure['conv_layers'][i]
            model.add(Conv2D(filters=conv_params['filters'],
                            kernel_size=conv_params['kernel_size'],
                            activation=conv_params['activation']))
            
            # Add pooling layer if available
            if pool_idx < len(structure['pool_layers']):
                pool_params = structure['pool_layers'][pool_idx]
                model.add(MaxPooling2D(pool_size=pool_params['pool_size']))
                pool_idx += 1
        
        # Add remaining pooling layers
        while pool_idx < len(structure['pool_layers']):
            pool_params = structure['pool_layers'][pool_idx]
            model.add(MaxPooling2D(pool_size=pool_params['pool_size']))
            pool_idx += 1
        
        # Flatten before fully connected layers
        model.add(Flatten())
        
        # Add fully connected layers
        for fc_params in structure['fc_layers']:
            model.add(Dense(units=fc_params['units'],
                           activation=fc_params['activation']))
            if fc_params['dropout'] > 0:
                model.add(Dropout(fc_params['dropout']))
        
        # Add output layer (binary classification)
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    def calculate_fitness(self, model, X_train, y_train, X_test, y_test, epochs=3):
        """Train CNN and calculate fitness function (BER)"""
        # Train model
        history = model.fit(X_train, y_train, 
                          validation_data=(X_test, y_test),
                          epochs=epochs, 
                          batch_size=32,
                          verbose=0)
        
        # Evaluate model
        y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
        
        # Calculate confusion matrix values
        TP = np.sum((y_pred == 1) & (y_test == 1))
        FP = np.sum((y_pred == 1) & (y_test == 0))
        TN = np.sum((y_pred == 0) & (y_test == 0))
        FN = np.sum((y_pred == 0) & (y_test == 1))
        
        # Calculate Balanced Error Rate (BER)
        if (TP + FP + TN + FN) == 0:
            return 1.0  # Worst fitness if prediction fails
        
        BER = (FP + FN) / (TP + FP + TN + FN)
        
        # Return fitness (lower BER is better)
        return BER
    
    def optimize(self, X_train, y_train, X_test, y_test):
        """Run AVOA optimization process"""
        # Initialize population
        population = self.initialize_population()
        
        # Initialize fitness and structures
        structures = [self.create_cnn_structure(particle) for particle in population]
        fitness_values = np.ones(self.population_size)
        pbest = np.ones(self.population_size)
        pbest_positions = population.copy()
        pbest_structures = structures.copy()
        
        # Initialize global best
        gbest_idx = 0
        gbest_fitness = 1.0
        gbest_position = None
        gbest_structure = None
        
        # For tracking progress
        fitness_history = []
        
        # Main loop
        for iteration in range(1, self.max_iter + 1):
            print(f"Iteration {iteration}/{self.max_iter}")
            
            # Evaluate each particle
            for i in range(self.population_size):
                print(f"  Evaluating particle {i+1}/{self.population_size}")
                
                # Build and evaluate model
                model = self.build_cnn_model(structures[i])
                fitness = self.calculate_fitness(model, X_train, y_train, X_test, y_test)
                fitness_values[i] = fitness
                
                # Update personal best
                if fitness < pbest[i]:
                    pbest[i] = fitness
                    pbest_positions[i] = population[i].copy()
                    pbest_structures[i] = structures[i]
            
            # Sort fitness values to identify best vultures
            sorted_indices = np.argsort(fitness_values)
            best_vulture1_idx = sorted_indices[0]
            best_vulture2_idx = sorted_indices[1]
            
            # Update global best
            if fitness_values[best_vulture1_idx] < gbest_fitness:
                gbest_idx = best_vulture1_idx
                gbest_fitness = fitness_values[best_vulture1_idx]
                gbest_position = population[best_vulture1_idx].copy()
                gbest_structure = structures[best_vulture1_idx]
            
            # Track best fitness
            fitness_history.append(gbest_fitness)
            
            # Calculate vulture satisfaction parameter (F)
            rand = np.random.rand()
            F = (2 * rand + 1) * (1 - (iteration / self.max_iter))
            
            # Update vulture positions
            for i in range(self.population_size):
                # Skip the two best vultures
                if i == best_vulture1_idx or i == best_vulture2_idx:
                    continue
                
                # Randomly select best vulture to follow
                q = np.random.rand()
                best_vulture_idx = best_vulture1_idx if q < 0.5 else best_vulture2_idx
                best_vulture_pos = population[best_vulture_idx].copy()
                
                # Update position based on F value
                new_position = population[i].copy()
                
                if abs(F) >= 1:  # Exploration
                    p1 = np.random.rand()
                    p2 = np.random.rand()
                    p3 = np.random.rand()
                    p4 = np.random.rand()
                    
                    A = p1 * (2 * p2 - 1)
                    B = 2 * p3
                    C = 2 * p4
                    
                    D = abs(C * best_vulture_pos - population[i])
                    L = A * B * D
                    
                    if abs(A) >= 1:
                        new_position = np.random.rand(3) * (self.maxc - self.minc) + self.minc
                    else:
                        new_position = best_vulture_pos - L
                
                elif 0.5 < abs(F) < 1:  # Exploitation - approach best vulture
                    new_position = best_vulture_pos - abs(best_vulture_pos - population[i]) * np.random.rand()
                
                else:  # Exploitation - spiral movement
                    D_vulture = abs(best_vulture_pos - population[i])
                    r = np.random.rand()
                    l = np.random.uniform(-1, 1)
                    new_position = D_vulture * np.exp(l) * np.cos(2 * np.pi * l) + best_vulture_pos
                
                # Ensure values are within bounds and are integers
                new_position[0] = max(min(int(round(new_position[0])), self.maxc), self.minc)
                new_position[1] = max(min(int(round(new_position[1])), self.maxp), self.minp)
                new_position[2] = max(min(int(round(new_position[2])), self.maxf), self.minf)
                
                # Update population and structure
                population[i] = new_position
                structures[i] = self.create_cnn_structure(new_position)
            
            print(f"  Best fitness in iteration {iteration}: {gbest_fitness}")
        
        self.fitness_history = fitness_history
        self.best_structure = gbest_structure
        
        # Plot fitness history
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.max_iter + 1), fitness_history)
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness (BER)')
        plt.title('AVOA Optimization Progress')
        plt.grid(True)
        plt.savefig('results/avoa_fitness_history.png')
        
        return gbest_structure, gbest_fitness

def optimize_hyperparameters(best_structure, X_train, y_train, X_test, y_test):
    """Optimize hyperparameters using AVOA"""
    # This would be a similar implementation to the structure optimization
    # but focuses on hyperparameters within the best structure
    
    # For simplicity, we'll just return the best structure for now
    return best_structure

def save_model(model, filename='models/best_model.h5'):
    """Save trained model"""
    model.save(filename)
    print(f"Model saved to {filename}")

def load_and_evaluate_model(model_path, X_test, y_test):
    """Load and evaluate model"""
    model = tf.keras.models.load_model(model_path)
    
    # Evaluate model
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    
    # Calculate metrics
    TP = np.sum((y_pred == 1) & (y_test == 1))
    FP = np.sum((y_pred == 1) & (y_test == 0))
    TN = np.sum((y_pred == 0) & (y_test == 0))
    FN = np.sum((y_pred == 0) & (y_test == 1))
    
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    
    return accuracy, precision, recall, f1_score