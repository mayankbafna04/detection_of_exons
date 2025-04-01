from Bio import SeqIO
import numpy as np
from pathlib import Path
import os
from scipy import signal

def process_genbank(input_dir, output_dir):
    """Process all GenBank files in directory"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_sequences = []
    all_labels = []
    
    for gb_file in Path(input_dir).glob("*.gb"):
        for record in SeqIO.parse(gb_file, "genbank"):
            seq = str(record.seq).upper()
            labels = np.zeros(len(seq), dtype=int)
            
            # Updated Biopython feature handling
            for feat in record.features:
                if feat.type == "exon":
                    try:
                        # Handle both simple and compound locations
                        for part in feat.location.parts:
                            start = int(part.start)
                            end = int(part.end)
                            labels[start:end] = 1
                    except AttributeError:
                        # Simple location case
                        start = int(feat.location.start)
                        end = int(feat.location.end)
                        labels[start:end] = 1
                    except Exception as e:
                        print(f"Error processing feature: {e}")
                        continue
            
            # Window processing with proper overlap
            window_size = 512
            step_size = 256
            for i in range(0, len(seq)-window_size+1, step_size):
                window_seq = seq[i:i+window_size]
                window_labels = labels[i:i+window_size]
                
                if window_seq.count('N')/window_size > 0.1:
                    continue
                
                # Use majority voting for window label
                window_label = 1 if np.mean(window_labels) > 0.5 else 0
                
                # Feature extraction
                voss = voss_mapping(window_seq)
                features = mgwt_transform(voss)
                
                all_sequences.append(features)
                all_labels.append(window_label)
    
    # Save processed data
    np.save(os.path.join(output_dir, "sequences.npy"), np.array(all_sequences))
    np.save(os.path.join(output_dir, "labels.npy"), np.array(all_labels))
    print(f"Processed {len(all_sequences)} samples")

def voss_mapping(sequence):
    """Convert DNA sequence to Voss representation with tuple fix"""
    mapping = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1]
    }
    return np.array([mapping.get(base, [0, 0, 0, 0]) for base in sequence])

def mgwt_transform(voss_data, scales=[8, 16, 32]):
    """Modified Gabor Wavelet Transform with proper 1D handling"""
    transformed = []
    
    for scale in scales:
        t = np.arange(-scale, scale+1)
        # Create 1D Gabor wavelet
        gabor_real = np.exp(-(t**2)/(2*(scale/2)**2)) * np.cos(2*np.pi*t/3)
        
        scale_features = []
        for i in range(4):  # A, C, G, T channels
            nucleotide_signal = voss_data[:, i]
            filtered = signal.convolve(nucleotide_signal, gabor_real, mode='same')
            scale_features.append(np.abs(filtered)**2)
        
        # Combine nucleotide responses
        scale_response = np.sum(np.array(scale_features), axis=0)
        transformed.append(scale_response)
    
    return np.column_stack(transformed)
