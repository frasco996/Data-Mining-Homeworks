import pandas as pd
import hashlib
import numpy as np
import time

class ShinglingMinHashingLSH:
    def __init__(self, document, shingle_length=10, hash_function=hashlib.sha1):
        self.document = str(document)  # Convert to string
        self.shingle_length = shingle_length
        self.hash_function = hash_function
        self.shingles = self.generate_shingles()
        self.hash_set = self.hash_shingles()

    def generate_shingles(self):
        shingles = set()
        for i in range(len(self.document) - self.shingle_length + 1):
            shingle = self.document[i:i + self.shingle_length]
            shingles.add(shingle)
        return shingles

    def hash_shingles(self):
        hash_set = set()
        for shingle in self.shingles:
            hashed_shingle = self.hash_function(shingle.encode()).hexdigest()
            hash_set.add(hashed_shingle)
        return hash_set

class MinHashSignature:
    def __init__(self, sets, num_hashes):
        self.sets = sets
        self.num_hashes = num_hashes
        self.hash_functions = self.generate_hash_functions()
        self.minhash_matrix = self.generate_minhash_matrix()

    def hash_family(self, i):
        result_size = 8  # how many bytes we want back
        max_len = 20  # how long can our i be (in decimal)
        salt = str(i).zfill(max_len)[-max_len:]

        def hash_member(x):
            return hashlib.sha1((x + salt).encode()).digest()[:result_size]

        return hash_member

    def generate_hash_functions(self):
        hash_functions = [self.hash_family(i) for i in range(self.num_hashes)]
        return hash_functions

    def generate_minhash_matrix(self):
        num_sets = len(self.sets)
        num_hash_functions = len(self.hash_functions)

        # Initialize minhash matrix with infinity
        minhash_matrix = np.full((num_hash_functions, num_sets), np.inf)

        # Generate hash values for each set
        for i, hash_function in enumerate(self.hash_functions):
            for j, s in enumerate(self.sets):
                for item in s:
                    hash_value = int.from_bytes(hash_function(str(item)), byteorder='big')
                    minhash_matrix[i, j] = min(minhash_matrix[i, j], hash_value)

        return minhash_matrix

    def get_signature(self, set_index):
        return tuple(self.minhash_matrix[:, set_index])

    def jaccard_similarity(self, set_index1, set_index2):
        signature_set1 = set(self.get_signature(set_index1))
        signature_set2 = set(self.get_signature(set_index2))
        intersection_size = len(signature_set1.intersection(signature_set2))
        union_size = len(signature_set1.union(signature_set2))
        return intersection_size / union_size if union_size != 0 else 0

class LocalitySensitiveHashing:
    def __init__(self, minhash_signature, threshold, bands, rows_per_band):
        self.minhash_signature = minhash_signature
        self.threshold = threshold
        self.bands = bands
        self.rows_per_band = rows_per_band

    def hash_band(self, band):
        return hash(tuple(band))

    def find_near_duplicates(self):
        num_sets = len(self.minhash_signature.sets)
        near_duplicates = set()

        for b in range(self.bands):
            band_start = b * self.rows_per_band
            band_end = (b + 1) * self.rows_per_band

            band_hashes = {}

            for i in range(num_sets):
                band = self.minhash_signature.minhash_matrix[band_start:band_end, i].tolist()
                band_hash = self.hash_band(band)

                if band_hash in band_hashes:
                    candidates = band_hashes[band_hash]
                    for candidate in candidates:
                        similarity = self.minhash_signature.jaccard_similarity(i, candidate)
                        if similarity >= self.threshold and i != candidate:
                            near_duplicates.add((i, candidate, similarity))
                else:
                    band_hashes[band_hash] = [i]

        return near_duplicates
    
class NonLocalitySensitiveHashing:
    def __init__(self, minhash_signature, threshold):
        self.minhash_signature = minhash_signature
        self.threshold = threshold

    def find_near_duplicates(self):
        num_sets = len(self.minhash_signature.sets)
        near_duplicates = set()

        for i in range(num_sets):
            for j in range(num_sets):
                similarity = self.minhash_signature.jaccard_similarity(i, j)
                if similarity >= self.threshold and i != j:
                    #print(self.minhash_signature.sets[i])
                    near_duplicates.add((i, j, similarity))

        return near_duplicates

print("Computing the LSH...")
df = pd.read_csv('./amazon_productsClean.tsv', sep='\t')

df["Hashed Shingles"] = df["Product Description"].apply(lambda x: ShinglingMinHashingLSH(x).hash_set)

shingled_sets = df["Hashed Shingles"].tolist()

threshold = 0.8

# Initialize MinHashSignature
minhash_signature100 = MinHashSignature(shingled_sets,100)
num_bands = 20
rows_per_band = 5

# Apply Locality-Sensitive Hashing
start_time_lsh1 = time.time()
nlsh = NonLocalitySensitiveHashing(minhash_signature100, threshold)
near_duplicates1 = nlsh.find_near_duplicates()
end_time_lsh1 = time.time()

for pair in near_duplicates1:
    print(f"Text: {df['Product Description'][pair[0]]}\nText {df['Product Description'][pair[1]]}\nIndex firt element: {pair[0]} \nIndex second element: {pair[1]} \nThey are near duplicates with Jaccard similarity {pair[2]}\n ")



#minhash_signature100 = MinHashSignature(shingled_sets,100)

# Apply Locality-Sensitive Hashing
start_time_lshh100 = time.time()

lsh = LocalitySensitiveHashing(minhash_signature100, threshold, num_bands, rows_per_band)
near_duplicates100 = lsh.find_near_duplicates()
end_time_lsh100 = time.time()

for pair in near_duplicates100:
    print(f"Text: {df['Product Description'][pair[0]]}\nText {df['Product Description'][pair[1]]}\nIndex firt element: {pair[0]} \nIndex second element: {pair[1]} \nThey are near duplicates with Jaccard similarity {pair[2]}\n ")


print("\nResults with no LSH:")
print("Number of near duplicates:", len(near_duplicates1))
print(f"\nTime taken with no LSH: {end_time_lsh1 - start_time_lsh1:.4f} seconds")


print("\nResults with LSH :")
print("Number of near duplicates:", len(near_duplicates100))
print(f"\nTime taken LSH: {end_time_lsh100 - start_time_lshh100:.4f} seconds")