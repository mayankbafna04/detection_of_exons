from Bio import Entrez
from Bio import SeqIO
import os

# Provide your email (NCBI requirement)
Entrez.email = "mayankbafna04@gmail.com"

accessions = [
# Multi-Exon Genes (238)
    "Z49258", "X73428", "X66114", "X14487", "X64467", "X01392", "X04898",
    "X01038", "X05151", "X52150", "X63600", "X68793", "X69907", "X72861",
    "X12706", "X04143", "Y00081", "X54486", "X52889", "X06882", "X14974",
    "X06180", "X15334", "X57152", "Z26491", "X62891", "X52851", "X14720",
    "Z18859", "Z46254", "X68303", "X02612", "X78212", "X84707", "X15215",
    "X79198", "Z48950", "X76776", "X02882", "X00492", "X61755", "Y00371",
    "X60459", "V00536", "Z00010", "X00695", "X03833", "X04500", "X64532",
    "V00565", "X03072", "X14445", "X52138", "X04981", "X62654", "Y00477",
    "X54489", "Z14977", "Z48051", "Z33457", "Z29373", "Y00067", "X16277",
    "X74614", "X54156", "X69438", "X75755", "X07881", "X70872", "X62025",
    "Z23102", "U08198", "U09954", "U12421", "U17969", "U19765", "U19816",
    "U20325", "U20499", "U20982", "U23143", "U23853", "U25826", "U26425",
    "X56997", "X76498", "X69953", "D25234", "K02212", "M21540", "M10277",
    "M13792", "M74493", "M16110", "M61831", "J04809", "M63420", "K02043",
    "J04982", "M14642", "M20902", "M10065", "J05096", "D28126", "M27132",
    "M64554", "M38180", "K01884", "D26535", "M27138", "J05008", "J04617",
    "D13156", "L29766", "M18079", "K02402", "L13391", "L24498", "L10822",
    "K01254", "L36861", "J03059", "M93280", "M91463", "M58600", "L04132",
    "M21339", "M83665", "L17131", "M26434", "M81806", "L39370", "M27024",
    "L35485", "L05072", "L14075", "L19546", "M23442", "L39064", "L33842",
    "J05253", "D16583", "M22638", "L11016", "M37719", "D49493", "J04718",
    "D90084", "L25648", "J05073", "L12760", "M27903", "L39891", "M11726",
    "M31951", "M30838", "J00301", "L34219", "D00591", "L11910", "M32405",
    "M89796", "M96759", "M18000", "M81650", "J00306", "M24461", "M64231",
    "M32639", "D32046", "L13470", "M96955", "J02846", "M11749", "M59924",
    "K03021", "L14927", "M37984", "D00596", "M21024", "J03589", "M73255",
    
    # Single-Exon Genes (142)
    "X57436", "X80536", "X14672", "X65633", "X63128", "X66503", "Z11162",
    "Y00106", "X52473", "V01511", "X68790", "X65784", "X55039", "Z25587",
    "X70251", "X60542", "X60382", "X55545", "X55760", "X62421", "X68302",
    "X12794", "X16545", "X55741", "X80915", "X66310", "Z23091", "X57130",
    "X57127", "X60481", "X65858", "X63337", "X64994", "X64995", "X03473",
    "X76786", "X00089", "X15265", "V00532", "X55293", "X52560", "X60201",
    "Z11901", "X12458", "X73424", "V00571", "X05246", "X13556", "X83416",
    "X53065", "X79235", "Z27113", "X52259", "X52075", "X71135", "X82554",
    "X55543", "X73534", "X82676", "U01212", "U03486", "U03735", "U10116",
    "U10273", "U10360", "U10554", "U11424", "U13666", "U13695", "U16812",
    "U17894", "U18548", "U20734", "U21051", "U22346", "L10381", "D13538",
    "L19704", "M11567", "L18972", "L37019", "M35160", "M27394", "M90355",
    "M90356", "M92269", "M31423", "J00119", "L15296", "M28170", "M14333",
    "L35240", "M60119", "M90439", "M55267", "M60830", "L10820", "D16826",
    "M69199", "J04152", "M86522", "M16514", "M22403", "M80478", "L36149",
    "M60094", "M97508", "M64799", "D29685", "M22005", "M26685"
]

# Create directories to store results
os.makedirs("genbank_files", exist_ok=True)
os.makedirs("cds_sequences", exist_ok=True)

def download_genbank_records(accessions):
    """Download GenBank records for given accessions"""
    for acc in accessions:
        try:
            print(f"Downloading {acc}...")
            # Fetch GenBank record
            handle = Entrez.efetch(
                db="nucleotide", 
                id=acc, 
                rettype="gb", 
                retmode="text"
            )
            
            # Save to file
            filename = f"genbank_files/{acc}.gb"
            with open(filename, "w") as f:
                f.write(handle.read())
            
            print(f"Saved {filename}")
            
        except Exception as e:
            print(f"Error downloading {acc}: {str(e)}")

def extract_cds_sequences(accessions):
    """Extract CDS sequences from downloaded GenBank files"""
    for acc in accessions:
        try:
            filename = f"genbank_files/{acc}.gb"
            if not os.path.exists(filename):
                print(f"File not found: {filename}")
                continue
                
            print(f"\nProcessing {acc}...")
            record = SeqIO.read(filename, "genbank")
            
            # Find all CDS features
            cds_count = 0
            for feature in record.features:
                if feature.type == "CDS":
                    cds_count += 1
                    # Extract CDS sequence
                    cds_seq = feature.extract(record.seq)
                    
                    # Get protein translation if available
                    translation = feature.qualifiers.get("translation", [""])[0]
                    
                    # Get product name
                    product = feature.qualifiers.get("product", ["unknown"])[0]
                    
                    # Save CDS to file
                    cds_filename = f"cds_sequences/{acc}_CDS_{cds_count}.fasta"
                    with open(cds_filename, "w") as f:
                        f.write(f">{acc}_CDS_{cds_count} {product}\n")
                        f.write(f"{cds_seq}\n")
                    
                    # Save protein translation if available
                    if translation:
                        protein_filename = f"cds_sequences/{acc}_PROTEIN_{cds_count}.fasta"
                        with open(protein_filename, "w") as f:
                            f.write(f">{acc}_PROTEIN_{cds_count} {product}\n")
                            f.write(f"{translation}\n")
                    
                    print(f"  CDS {cds_count}: {feature.location}")
                    print(f"  Product: {product}")
                    print(f"  Sequence length: {len(cds_seq)} bp")
                    if translation:
                        print(f"  Protein length: {len(translation)} aa")
            
            if cds_count == 0:
                print(f"  No CDS features found in {acc}")
                
        except Exception as e:
            print(f"Error processing {acc}: {str(e)}")

# Run the pipeline
if __name__ == "__main__":
    print("Starting GenBank download...")
    download_genbank_records(accessions)
    
    print("\nExtracting CDS sequences...")
    extract_cds_sequences(accessions)
    
    print("\nProcessing complete!")
