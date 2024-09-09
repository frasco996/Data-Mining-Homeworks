import hashlib
import numpy as np


class MinHashSignature:
    def __init__(self, sets, num_hashes=400):
        self.sets = sets
        self.num_hashes = num_hashes
        self.hash_functions = self.generate_hash_functions()
        self.minhash_matrix = self.generate_minhash_matrix()

    def hash_family(self, i):
        result_size = 8  # how many bytes we want back
        max_len = 20  # how long can our i be (in decimal)
        salt = str(i).zfill(max_len)[-max_len:]

        def hash_member(x):
            return hashlib.sha256((x + salt).encode()).digest()[:result_size]

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

# Example usage:
sets = [{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28}]

minhash_signature = MinHashSignature(sets)

# Get the MinHash signature for the first set
signature_set1 = minhash_signature.get_signature(0)

# Get the MinHash signature for the second set
signature_set2 = minhash_signature.get_signature(1)

# Calculate Jaccard similarity between sets 1 and 2
jaccard_similarity = minhash_signature.jaccard_similarity(0, 1)

# Print the MinHash signatures and Jaccard similarity
print("MinHash Signature for Set 1:", signature_set1)
print("MinHash Signature for Set 2:", signature_set2)
print("\nJaccard Similarity between Set 1 and Set 2:", jaccard_similarity)
print("\n")