#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.sparse import csc_matrix
import timeit
import concurrent.futures
import matplotlib.pyplot as plt


# In[2]:


def random_projections(data_matrix, num_permutations):
    np.random.seed(172)
    random_perm_matrix = np.random.choice([-1, 1], size=(num_permutations, data_matrix.shape[0]))
    signature_matrix = np.zeros((random_perm_matrix.shape[0], data_matrix.shape[1]))

    for c in range(data_matrix.shape[1]):
        x_T = data_matrix[:, c].T
        for r in range(num_permutations): 
            dot_product = x_T.dot(random_perm_matrix[r, :])

            if dot_product > 0:
                signature_matrix[r, c] = 1
            else:
                signature_matrix[r, c] = -1

    return signature_matrix


# In[3]:


def cosine_similarity(p1, p2):
    mag_p1 = np.linalg.norm(p1)
    mag_p2 = np.linalg.norm(p2)
    
    # Check for almost zero vectors
    if mag_p1 < 1e-10 or mag_p2 < 1e-10:
        return 1.0
    
    dot_product = np.dot(p1, p2)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Clip values to ensure they are within valid range
    
    cos_dist = np.arccos(dot_product / (mag_p1 * mag_p2))
    cos_sim = 1 - cos_dist / np.pi
        
    return cos_sim


# In[4]:


def random_hash_functions(num_functions, num_buckets):
    np.random.seed(1702)
    hash_functions = []
    for _ in range(num_functions):
        a = np.random.randint(1, num_buckets)
        b = np.random.randint(0, num_buckets)
        hash_functions.append(lambda x: tuple((a * xi + b) % num_buckets for xi in x))
    return hash_functions


# In[5]:


def apply_lsh_band(band_id, signature_matrix, hash_functions, rows_per_band, num_users):
    start_row = band_id * rows_per_band
    end_row = (band_id + 1) * rows_per_band
    sub_matrix = signature_matrix[start_row:end_row, :]

    bucket_list = {}

    for col_id in range(num_users):
        hashed_value = tuple(hf(tuple(sub_matrix[:, col_id])) for hf in hash_functions)
        if hashed_value not in bucket_list:
            bucket_list[hashed_value] = []
        bucket_list[hashed_value].append(col_id)

    return bucket_list


# In[6]:


def apply_lsh_parallel(signature_matrix, hash_functions, num_bands, rows_per_band):
    num_users = signature_matrix.shape[1]
    bucket_lists = [{} for _ in range(num_bands)]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        band_results = list(executor.map(
            lambda x: apply_lsh_band(x, signature_matrix, hash_functions, rows_per_band, num_users),
            range(num_bands)
        ))

    for band_id, band_result in enumerate(band_results):
        bucket_lists[band_id] = band_result

    return bucket_lists


# In[7]:


def process_column(col_id, sub_matrix, hash_functions):
    return tuple(hf(tuple(sub_matrix[:, col_id])) for hf in hash_functions)


# In[8]:


def find_candidate_pairs(bucket_lists, cosine_similarity_threshold):
    candidate_pairs = set()

    for buckets in bucket_lists:
        for bucket_id, user_list in buckets.items():
            if len(user_list) > 1:
                for i in range(len(user_list)):
                    for j in range(i + 1, len(user_list)):
                        user1 = user_list[i]
                        user2 = user_list[j]
                        candidate_pairs.add(tuple(sorted([user1, user2])))

    return candidate_pairs


# In[9]:


def write_to_file(file_path, pairs):
    with open('cs.txt', 'w') as f:  # Use 'a' mode to append to the file
        for pair in pairs:
            f.write(f"{pair[0]}, {pair[1]}\n")


# In[10]:


def calculate_cosine_similarity(pair, sparse_col_matrix):
    u1, u2 = pair
    v1 = sparse_col_matrix[:, u1].toarray().flatten()
    v2 = sparse_col_matrix[:, u2].toarray().flatten()

    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)

    if mag_v1 == 0 or mag_v2 == 0:
        return u1, u2, 0  # Avoid division by zero

    cos_dist = np.arccos(np.dot(v1, v2) / (mag_v1 * mag_v2))
    cos_sim = 1 - cos_dist / np.pi

    return u1, u2, cos_sim


# In[12]:


def main():
    # Load the full data
    data = np.load('user_movie_rating.npy')
    
    # Create a sparse column matrix
    sparse_col_matrix = csc_matrix((data[:, 2], (data[:, 1], data[:, 0])))

    # Create a signature matrix
    print("Create signature matrix...")
    begin = timeit.default_timer()
    signature_matrix = random_projections(sparse_col_matrix, num_permutations=80)
    end = timeit.default_timer() - begin
    print('Time to make signature matrix: ', end / 60, ' minutes')

    # Define LSH parameters
    num_hash_functions = 10
    num_buckets = 1000000
    num_bands = 2  # Experiment with the number of bands
    rows_per_band = 21  # Experiment with the number of rows per band
    #These values of (b=2, r=21) give us most pairs while remaining < 10 min on a 8 Gb RAM computer

    # Generate random hash functions
    hash_functions = random_hash_functions(num_hash_functions, num_buckets)

    # Apply LSH in parallel
    begin = timeit.default_timer()
    bucket_lists = apply_lsh_parallel(signature_matrix, hash_functions, num_bands, rows_per_band)
    end = timeit.default_timer() - begin
    print('LSH Execution Time: ', end / 60, ' minutes')


    # Find candidate pairs
    begin = timeit.default_timer()
    candidate_pairs = find_candidate_pairs(bucket_lists,cosine_similarity_threshold=0.73)
    end = timeit.default_timer() - begin
    print('Candidate Pairs Execution Time: ', end / 60, ' minutes')
    print(len(candidate_pairs))

    # Convert candidate_pairs set to a list
    candidate_pairs = list(candidate_pairs)
    begin = timeit.default_timer()
    # Calculate cosine similarity for candidate pairs
    final_pairs_cs = []

    def process_batch(batch):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda p: calculate_cosine_similarity(p, sparse_col_matrix), batch))
        return [pair for pair in results if pair[2] > 0.73]

    batch_size = 5000
    for i in range(0, len(candidate_pairs), batch_size):
        batch = candidate_pairs[i:i + batch_size]
        final_pairs_cs.extend(process_batch(batch))
    end = timeit.default_timer() - begin
    # Dump results to files
    write_to_file('cosine_sim_results.txt', final_pairs_cs)
    print('Final Pairs Execution Time: ', end / 60, ' minutes')
    print("Cosine Similarity:", len(final_pairs_cs), "final pairs:\n", final_pairs_cs)
        
    
    # Scatter plot for most similar pairs
    x_data = np.arange(1, len(final_pairs_cs)+1)
    y_data = np.array([])
    for i in range(0, len(final_pairs_cs)):
        y_data = np.append(y_data, final_pairs_cs[i][2])
    y_data = np.sort(y_data)

    plt.scatter(x_data, y_data, s=5)
    plt.title('%i pairs with CS > 0.5' %len(x_data))
    plt.xlabel('Most similar pairs')
    plt.ylabel('Cosine Similarity')
    plt.show()

    
    return final_pairs_cs, signature_matrix

final_pairs_cs, signature_matrix = main()


# In[ ]:





# In[ ]:




