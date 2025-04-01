import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Flatten, BatchNormalization
import copy
from sklearn.model_selection import train_test_split

class AVOAOptimizer:
    def __init__(self, pop_size=30, max_iter=20):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.best_config = None
        self.min_layers = 5
        self.max_layers = 15
        self.max_filters = 512
        
        # AVOA parameters
        self.p1 = 0.7  # Exploration probability
        self.alpha = 0.2  # Random walk factor

    class CNNIndividual:
        def __init__(self):
            self.layers = []
            self.fitness = 0.0
            
        def create_model(self, input_shape):
            """Create CNN model with proper kernel_size tuples"""
            model = Sequential()
            model.add(tf.keras.layers.Input(shape=input_shape))
            
            # Ensure valid architecture
            self._validate_layers()
            
            for layer in self.layers:
                if layer['type'] == 'conv':
                    model.add(Conv1D(
                        filters=layer['filters'],
                        kernel_size=layer['kernel_size'],
                        activation='relu',
                        padding='same'
                    ))
                    model.add(BatchNormalization())
                elif layer['type'] == 'pool':
                    model.add(MaxPool1D(
                        pool_size=layer['pool_size']
                    ))
            
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            return model
        
        def _validate_layers(self):
            """Ensure all layer parameters are tuples"""
            for layer in self.layers:
                if layer['type'] == 'conv' and not isinstance(layer['kernel_size'], tuple):
                    layer['kernel_size'] = (layer['kernel_size'],)
                elif layer['type'] == 'pool' and not isinstance(layer['pool_size'], tuple):
                    layer['pool_size'] = (layer['pool_size'],)

    def initialize_population(self):
        """Generate initial population with valid architectures"""
        population = []
        for _ in range(self.pop_size):
            ind = self.CNNIndividual()
            num_layers = np.random.randint(self.min_layers, self.max_layers+1)
            
            for _ in range(num_layers):
                if np.random.rand() < 0.7:  # 70% chance for conv layer
                    layer = {
                        'type': 'conv',
                        'filters': np.random.choice([32, 64, 128, 256]),
                        'kernel_size': (np.random.choice([3, 5, 7]),)
                    }
                else:  # 30% chance for pooling layer
                    layer = {
                        'type': 'pool',
                        'pool_size': (np.random.choice([2, 3, 4]),)
                    }
                ind.layers.append(layer)
            
            self._validate_architecture(ind)
            population.append(ind)
        return population

    def _validate_architecture(self, individual):
        """Ensure valid CNN architecture"""
        # 1. Ensure first layer is convolutional
        if individual.layers[0]['type'] != 'conv':
            individual.layers.insert(0, {
                'type': 'conv',
                'filters': 32,
                'kernel_size': (3,)
            })
        
        # 2. Prevent consecutive pooling layers
        new_layers = []
        prev_type = None
        for layer in individual.layers:
            if prev_type == 'pool' and layer['type'] == 'pool':
                # Insert conv layer between pooling layers
                new_layers.append({
                    'type': 'conv',
                    'filters': np.random.choice([32, 64, 128, 256]),
                    'kernel_size': (3,)
                })
            new_layers.append(layer)
            prev_type = layer['type']
        individual.layers = new_layers
        
        # 3. Ensure minimum layers
        while len(individual.layers) < self.min_layers:
            individual.layers.append({
                'type': 'conv',
                'filters': np.random.choice([32, 64, 128, 256]),
                'kernel_size': (3,)
            })

    def calculate_fitness(self, individual, X, y):
        """Calculate fitness with proper validation split"""
        try:
            # Split training and validation data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            model = individual.create_model(X_train.shape[1:])
            history = model.fit(
                X_train, y_train,
                epochs=20,
                batch_size=64,
                validation_data=(X_val, y_val),
                verbose=0
            )
            
            # Calculate BER on validation set
            y_pred = (model.predict(X_val) > 0.5).astype(int).flatten()
            fn = np.sum((y_val == 1) & (y_pred == 0))
            fp = np.sum((y_val == 0) & (y_pred == 1))
            ber = (fp + fn) / len(y_val)
            return 1 - ber
        except Exception as e:
            print(f"Invalid architecture: {str(e)}")
            return 0.0

    def avoa_step(self, population, best_index):
        """Optimization step with architecture validation"""
        new_pop = []
        best = population[best_index]
        
        for i in range(self.pop_size):
            # Mutation operations
            if np.random.rand() < self.p1:
                new_ind = self._random_walk(population[i])
            else:
                new_ind = self._follow_best(population[i], best)
            
            self._validate_architecture(new_ind)
            new_pop.append(new_ind)
        
        return new_pop

    def _random_walk(self, individual):
        """Mutation with tuple preservation"""
        new_ind = copy.deepcopy(individual)
        idx = np.random.randint(0, len(new_ind.layers))
        
        if new_ind.layers[idx]['type'] == 'conv':
            # Mutate conv layer
            new_ind.layers[idx]['filters'] = np.clip(
                new_ind.layers[idx]['filters'] + np.random.randint(-32, 32),
                32, 256
            )
            new_ind.layers[idx]['kernel_size'] = (np.random.choice([3, 5, 7]),)
        else:
            # Mutate pool layer
            new_ind.layers[idx]['pool_size'] = (np.random.choice([2, 3, 4]),)
        
        return new_ind

    def _follow_best(self, current, best):
        """Crossover operation with architecture validation"""
        new_ind = copy.deepcopy(current)
        min_len = min(len(new_ind.layers), len(best.layers))
        
        for i in range(min_len):
            if np.random.rand() < 0.5:
                new_ind.layers[i] = copy.deepcopy(best.layers[i])
        
        return new_ind

    def optimize(self, X, y):
        """Main optimization loop with safety checks"""
        population = self.initialize_population()
        best_fitness = 0.0
        self.best_config = population[0]  # Initialize with first individual
        
        for iter_num in range(self.max_iter):
            fitness = []
            valid_individuals = 0
            
            for ind in population:
                score = self.calculate_fitness(ind, X, y)
                if score > 0:  # Count valid architectures
                    valid_individuals += 1
                fitness.append(score)
                
                if score > best_fitness:
                    best_fitness = score
                    self.best_config = copy.deepcopy(ind)
            
            if valid_individuals == 0:
                raise RuntimeError("All architectures are invalid. Check constraints.")
            
            best_idx = np.argmax(fitness)
            print(f"Iteration {iter_num+1}: Best BER = {1 - best_fitness:.4f}")
            
            population = self.avoa_step(population, best_idx)
        
        return self.best_config
