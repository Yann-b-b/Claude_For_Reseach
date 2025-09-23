import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import requests
import random
from io import StringIO

def download_bacterial_genomes():
    """Download representative bacterial genome sequences"""

    # Common bacteria in antimicrobial studies with their NCBI accession numbers
    bacterial_genomes = {
        'E. coli': 'NC_000913.3',  # E. coli K-12 MG1655
        'S. aureus': 'NC_007795.1',  # S. aureus NCTC 8325
        'B. subtilis': 'NC_000964.3',  # B. subtilis 168
        'P. aeruginosa': 'NC_002516.2',  # P. aeruginosa PAO1
        'S. pneumoniae': 'NC_003028.3',  # S. pneumoniae TIGR4
    }

    genome_sequences = {}

    for species, accession in bacterial_genomes.items():
        print(f"Downloading {species} genome...")

        # Use NCBI E-utilities to fetch the sequence
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={accession}&rettype=fasta&retmode=text"

        try:
            response = requests.get(url)
            if response.status_code == 200:
                # Extract sequence from FASTA format
                lines = response.text.strip().split('\n')
                sequence = ''.join(lines[1:])  # Skip header line
                genome_sequences[species] = sequence
                print(f"Successfully downloaded {species} genome ({len(sequence):,} bp)")
            else:
                print(f"Failed to download {species} genome")
                # Use a random sequence as fallback
                genome_sequences[species] = generate_random_dna_sequence(1000000)

        except Exception as e:
            print(f"Error downloading {species}: {e}")
            # Use a random sequence as fallback
            genome_sequences[species] = generate_random_dna_sequence(1000000)

    return genome_sequences

def generate_random_dna_sequence(length):
    """Generate a random DNA sequence"""
    bases = ['A', 'T', 'G', 'C']
    return ''.join(random.choices(bases, k=length))

def map_bacteria_to_species(bacterium_name):
    """Map various bacterial naming conventions to standard species"""
    bacterium_name = bacterium_name.lower().strip()

    # Create mapping for common variations
    species_mapping = {
        'e. coli': 'E. coli',
        'escherichia coli': 'E. coli',
        'ecoli': 'E. coli',
        's. aureus': 'S. aureus',
        'staphylococcus aureus': 'S. aureus',
        'staph aureus': 'S. aureus',
        'b. subtilis': 'B. subtilis',
        'bacillus subtilis': 'B. subtilis',
        'p. aeruginosa': 'P. aeruginosa',
        'pseudomonas aeruginosa': 'P. aeruginosa',
        's. pneumoniae': 'S. pneumoniae',
        'streptococcus pneumoniae': 'S. pneumoniae',
        's. pneumonia': 'S. pneumoniae',  # Common misspelling
        'strep pneumoniae': 'S. pneumoniae',
    }

    # Try direct mapping first
    if bacterium_name in species_mapping:
        return species_mapping[bacterium_name]

    # Try partial matching
    for key, value in species_mapping.items():
        if key in bacterium_name or bacterium_name in key:
            return value

    # Default to E. coli for unmapped bacteria (most common model organism)
    return 'E. coli'

def extract_dna_subsequence(genome_sequence, target_length=1000):
    """Extract a random subsequence from a bacterial genome"""
    if len(genome_sequence) <= target_length:
        return genome_sequence

    start_pos = random.randint(0, len(genome_sequence) - target_length)
    return genome_sequence[start_pos:start_pos + target_length]

def process_grampa_dataset():
    """Process the GRAMPA dataset and prepare it for training"""

    print("Loading GRAMPA dataset...")
    df = pd.read_csv('grampa_dataset.csv')

    print(f"Original dataset size: {len(df)} entries")

    # Remove entries with missing critical data
    df = df.dropna(subset=['bacterium', 'sequence', 'value'])
    print(f"After removing missing data: {len(df)} entries")

    # Clean and prepare the data
    df['bacterium'] = df['bacterium'].str.strip()
    df['sequence'] = df['sequence'].str.strip()

    # Remove entries with very short or very long sequences
    df = df[(df['sequence'].str.len() >= 5) & (df['sequence'].str.len() <= 200)]
    print(f"After filtering sequence length: {len(df)} entries")

    # Convert MIC values to binary labels (antimicrobial vs non-antimicrobial)
    # Use median value as threshold or use a biological meaningful threshold
    threshold = df['value'].median()  # or use a fixed threshold like 0
    df['antimicrobial_activity'] = (df['value'] > threshold).astype(int)

    print(f"Using threshold: {threshold:.3f}")
    print(f"Positive samples: {df['antimicrobial_activity'].sum()} ({df['antimicrobial_activity'].mean()*100:.1f}%)")

    return df

def create_training_dataset():
    """Create the final training dataset with bacterial DNA and peptide sequences"""

    # Download bacterial genomes
    print("=== Downloading Bacterial Genomes ===")
    genome_sequences = download_bacterial_genomes()

    # Process GRAMPA dataset
    print("\n=== Processing GRAMPA Dataset ===")
    df = process_grampa_dataset()

    # Map bacteria names to standard species and get DNA sequences
    print("\n=== Mapping Bacterial DNA Sequences ===")
    dna_sequences = []
    protein_sequences = []
    labels = []
    bacterium_species = []

    for idx, row in df.iterrows():
        # Map bacterium name to standard species
        species = map_bacteria_to_species(row['bacterium'])

        # Get corresponding genome sequence
        if species in genome_sequences:
            genome = genome_sequences[species]
        else:
            genome = genome_sequences['E. coli']  # Default fallback

        # Extract a random subsequence from the bacterial genome
        dna_subseq = extract_dna_subsequence(genome, target_length=1000)

        dna_sequences.append(dna_subseq)
        protein_sequences.append(row['sequence'])
        labels.append(row['antimicrobial_activity'])
        bacterium_species.append(species)

        if len(dna_sequences) % 5000 == 0:
            print(f"Processed {len(dna_sequences)} sequences...")

    print(f"\nFinal dataset: {len(dna_sequences)} entries")

    # Create DataFrame with final data
    final_df = pd.DataFrame({
        'bacterium_species': bacterium_species,
        'dna_sequence': dna_sequences,
        'protein_sequence': protein_sequences,
        'antimicrobial_activity': labels
    })

    # Show species distribution
    print("\nSpecies distribution:")
    print(final_df['bacterium_species'].value_counts())

    print(f"\nClass distribution:")
    print(final_df['antimicrobial_activity'].value_counts())
    print(f"Positive class percentage: {final_df['antimicrobial_activity'].mean()*100:.1f}%")

    # Save the processed dataset
    print("\nSaving processed dataset...")
    final_df.to_csv('antimicrobial_training_data.csv', index=False)
    print("Dataset saved as 'antimicrobial_training_data.csv'")

    # Create separate files for the training script
    print("\nCreating training files...")

    # Extract sequences and labels for training
    dna_seqs = final_df['dna_sequence'].tolist()
    protein_seqs = final_df['protein_sequence'].tolist()
    labels_list = final_df['antimicrobial_activity'].tolist()

    return dna_seqs, protein_seqs, labels_list

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create the training dataset
    dna_sequences, protein_sequences, labels = create_training_dataset()

    print(f"\n=== Dataset Summary ===")
    print(f"Total sequences: {len(dna_sequences)}")
    print(f"DNA sequence length: {len(dna_sequences[0])} bp")
    print(f"Average protein length: {np.mean([len(seq) for seq in protein_sequences]):.1f} aa")
    print(f"Positive samples: {sum(labels)} ({np.mean(labels)*100:.1f}%)")

    # Show sample data
    print(f"\n=== Sample Data ===")
    for i in range(3):
        print(f"Sample {i+1}:")
        print(f"  DNA length: {len(dna_sequences[i])} bp")
        print(f"  DNA sequence: {dna_sequences[i][:50]}...")
        print(f"  Protein sequence: {protein_sequences[i]}")
        print(f"  Label: {labels[i]} ({'Antimicrobial' if labels[i] else 'Not antimicrobial'})")
        print()

    print("Dataset preparation complete! You can now run antimicrobial_predictor.py")