import numpy as np
from scipy.signal import gabor

class FeatureExtractor:
    def __init__(self, window_size=256, stride=100):
        self.window_size = window_size
        self.stride = stride

    def voss_mapping(self, sequence):
        """Convert DNA to 4-channel binary representation"""
        mapping = {
            'A': [1, 0, 0, 0],
            'T': [0, 1, 0, 0],
            'C': [0, 0, 1, 0],
            'G': [0, 0, 0, 1]
        }
        return np.array([mapping.get(base, [0,0,0,0]) for base in sequence])

    def mgwt_transform(self, signal):
        """Modified Gabor Wavelet Transform"""
        transformed = []
        for i in range(0, len(signal)-self.window_size, self.stride):
            segment = signal[i:i+self.window_size]
            # Apply Gabor filter at 1/3 frequency (for 3-base periodicity)
            real, _ = gabor(segment.flatten(), frequency=1/3)
            power = np.abs(real)**2
            transformed.append(power)
        return np.array(transformed)

    def create_dataset(self, sequences, exon_ranges):
        """Generate training samples with labels"""
        X, y = [], []
        for seq, exons in zip(sequences, exon_ranges):
            # Convert to Voss representation
            voss = self.voss_mapping(seq)
            
            # Create binary exon labels (1=exon, 0=intron)
            labels = np.zeros(len(seq))
            for start, end in exons:
                labels[start:end] = 1
            
            # Apply MGWT and create windows
            features = self.mgwt_transform(voss)
            for i, window in enumerate(features):
                X.append(window)
                # Get majority label for window
                window_labels = labels[i*self.stride : i*self.stride+self.window_size]
                y.append(1 if np.mean(window_labels) > 0.5 else 0)
        
        return np.array(X), np.array(y)