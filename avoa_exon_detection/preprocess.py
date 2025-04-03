import os
import numpy as np
import pandas as pd
from Bio import SeqIO
import pywt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def read_genbank_file(file_path):
    """Read genbank file and extract DNA sequence and exon locations"""
    records = list(SeqIO.parse(file_path, "genbank"))
    
    for record in records:
        sequence = str(record.seq)
        exon_locations = []
        
        for feature in record.features:
            if feature.type == "CDS":
                start = feature.location.start
                end = feature.location.end
                exon_locations.append((start, end))
        
        return sequence, exon_locations

def voss_mapping(sequence):
    """Convert DNA sequence to binary indicator sequences using Voss mapping"""
    seq_length = len(sequence)
    a_seq = np.zeros(seq_length)
    c_seq = np.zeros(seq_length)
    g_seq = np.zeros(seq_length)
    t_seq = np.zeros(seq_length)
    
    for i, nucleotide in enumerate(sequence.upper()):
        if nucleotide == 'A':
            a_seq[i] = 1
        elif nucleotide == 'C':
            c_seq[i] = 1
        elif nucleotide == 'G':
            g_seq[i] = 1
        elif nucleotide == 'T':
            t_seq[i] = 1
    
    return a_seq, c_seq, g_seq, t_seq

def modified_gabor_wavelet_transform(signal, omega0=2*np.pi/3):
    """Apply Modified Gabor Wavelet Transform to extract features"""
    # Parameters for wavelet transform
    scales = np.arange(1, 128)
    coeffs = []
    
    # Calculate wavelet coefficients
    for a in scales:
        # Create Gabor wavelet
        wavelet = lambda x: np.exp(-x**2/2) * np.exp(1j * omega0 * x)
        
        # Apply wavelet transform
        coef = np.zeros(len(signal), dtype=complex)
        for b in range(len(signal)):
            x_values = np.arange(len(signal))
            shifted_x = (x_values - b) / a
            phi = np.exp(-shifted_x**2/2) * np.exp(1j * omega0 * shifted_x)
            coef[b] = np.sum(signal * phi)
        
        coeffs.append(coef)
    
    # Calculate spectrum
    spectrum = np.abs(np.array(coeffs))**2
    
    # Get TBP components at frequency ω₀=N/3
    tbp_index = np.argmin(np.abs(scales - len(signal)/3))
    tbp_component = spectrum[tbp_index]
    
    return tbp_component

def prepare_frames(signals, labels, frame_length=256, overlap=100):
    """Prepare data frames for CNN"""
    frames = []
    frame_labels = []
    
    for i, signal in enumerate(signals):
        for j in range(0, len(signal) - frame_length, frame_length - overlap):
            frame = signal[j:j+frame_length]
            
            # Skip frames that are too short
            if len(frame) < frame_length:
                continue
                
            # Convert vector to 16x16 matrix
            if len(frame) > frame_length:
                frame = frame[:frame_length]
            
            # Reshape to matrix
            matrix = frame.reshape(16, 16)
            frames.append(matrix)
            frame_labels.append(labels[i])
    
    return np.array(frames), np.array(frame_labels)

def preprocess_data(file_path, output_dir="data"):
    """Preprocess the GenBank file and save features for CNN"""
    # Read data
    sequence, exon_locations = read_genbank_file(file_path)
    
    # Create binary mask for exons (1 for exon, 0 for intron)
    exon_mask = np.zeros(len(sequence))
    for start, end in exon_locations:
        exon_mask[start:end] = 1
    
    # Apply Voss mapping
    a_seq, c_seq, g_seq, t_seq = voss_mapping(sequence)
    
    # Apply MGWT to each binary sequence
    a_features = modified_gabor_wavelet_transform(a_seq)
    c_features = modified_gabor_wavelet_transform(c_seq)
    g_features = modified_gabor_wavelet_transform(g_seq)
    t_features = modified_gabor_wavelet_transform(t_seq)
    
    # Combine features
    combined_features = a_features + c_features + g_features + t_features
    
    # Normalize features
    normalized_features = (combined_features - np.min(combined_features)) / (np.max(combined_features) - np.min(combined_features))
    
    # Separate signals into exon and intron classes
    exon_signals = []
    intron_signals = []
    
    # Use sliding window to create segments
    window_size = 256
    stride = 128
    
    for i in range(0, len(normalized_features) - window_size, stride):
        window_features = normalized_features[i:i+window_size]
        window_mask = exon_mask[i:i+window_size]
        
        # Classify window as exon if >50% of positions are exons
        if np.mean(window_mask) > 0.5:
            exon_signals.append(window_features)
        else:
            intron_signals.append(window_features)
    
    # Prepare labels
    exon_labels = np.ones(len(exon_signals))  # Class 1 for exons
    intron_labels = np.zeros(len(intron_signals))  # Class 0 for introns
    
    # Combine signals and labels
    all_signals = np.concatenate([exon_signals, intron_signals])
    all_labels = np.concatenate([exon_labels, intron_labels])
    
    # Prepare frames
    frames, frame_labels = prepare_frames(all_signals, all_labels)
    
    # Reshape for CNN input (adding channel dimension)
    frames = frames.reshape(-1, 16, 16, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        frames, frame_labels, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Save preprocessed data
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Plot example of processed signal
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(normalized_features[:1000])
    plt.title('Normalized MGWT features')
    
    plt.subplot(2, 1, 2)
    plt.plot(exon_mask[:1000])
    plt.title('Exon mask (1=exon, 0=intron)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'signal_visualization.png'))
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Preprocess data
    genbank_file = "genbank_files"
    X_train, X_test, y_train, y_test = preprocess_data(genbank_file)