def voss_mapping(dna_sequence):
    mapping = {'A': [1,0,0,0], 'T': [0,1,0,0],
               'C': [0,0,1,0], 'G': [0,0,0,1]}
    return np.array([mapping[base] for base in dna_sequence])
