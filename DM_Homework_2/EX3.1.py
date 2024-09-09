import hashlib
import pandas as pd

class ShinglingMinHashingLSH:
    def __init__(self, document, shingle_length, hash_function=hashlib.sha1):
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
        #print(shingles)
        return shingles

    def hash_shingles(self):
        hash_set = set()
        for shingle in self.shingles:
            hashed_shingle = self.hash_function(shingle.encode()).hexdigest()
            hash_set.add(hashed_shingle)
        #print(hash_set)
        return hash_set

data = pd.DataFrame({"Product Description": ["This is an example document for shingling and minhashing.",
                                             "Another document that is made for me since two years ago"]})


k = input("Enter length of shingles: ")
data["Hashed Shingles"] = data["Product Description"].apply(lambda x: ShinglingMinHashingLSH(x,int(k)).hash_set)
# Display the DataFrame with hashed shingles
output_file_path = 'DataFrame.tsv'
data.to_csv(output_file_path, sep='\t', index=False)

print(data)
