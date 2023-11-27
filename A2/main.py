#Import all the neccessary libraries
import numpy as np
from scipy.sparse import csr_matrix,csc_matrix
import concurrent.futures
import timeit
import matplotlib.pyplot as plt
import sys, getopt

def loadData_js(data_path):
    """
    This function is dedicated to loading in the data, converting the data to a sparse row matrix (since the data is sparse), 
    and finally inspecting the data a bit to see whether the number of users and movies match up with our expectations

    Args:
        data_path::[str]
            The path to the data file, in this case user_movie_rating.npy
    Returns:
        sparse_rowMatrix_full::[sparse matrix]
            Returns a sparse row matrix containing all the data, with movieIDs as rows and userIDs as columns 
    """
    #Load in the large data file and inspect the data 
    data = np.load(data_path)

    #Split the data into three lists of ratings, movies and users respectively. 
    user_ids_full = data[:,0]
    movie_ids_full = data[:,1]
    ratings_full = data[:,2]

    #Creating a sparse row matrix, use the lowest data type (int8) possible in this case to reduce RAM cost 
    sparse_rowMatrix_full = csr_matrix((ratings_full, (movie_ids_full, user_ids_full)),dtype=np.int8)

    #Remove the first row and column of the sparse row matrix, since it adds one too many for each 
    sparse_rowMatrix_full = sparse_rowMatrix_full[:,1:]
    sparse_rowMatrix_full = sparse_rowMatrix_full[1:,:]
    
    #Check the amount of users
    print("Number of users is: ", np.shape(sparse_rowMatrix_full[0,:])[1])
    print("Number of movies is: ", np.shape(sparse_rowMatrix_full)[0])
    return sparse_rowMatrix_full

def minhashing(data_matrix,row_fraction,num_permutations,seed):
    """
    This function creates a signature matrix using the sparse row matrix as input. To do so, the Minhashing theory described
    in the report is used. For more information on the general picture, please consult the report. 
    The size of the produced signature matrix can be influenced by choosing the number of permutations, num_permutations. 

    Args:
        data_matrix::[sparse matrix]
            Uses a sparse row matrix created using the data. 
        row_fraction::[float]
            This argument tells the code what fraction of the number of movies to use in producing the signature matrix.
            Using this significantly speeds up the signature matrix creation.
        num_permutations::[int]
            Tells the code how many permutations to use for the signature matrix, directly corresponding to the number
            of rows the signature matrix will contain. 
        seed::[int]
            The NumPy seed used for the random functions.
        
    Returns:
        signatureMatrix::[NumPy 2D array]
            Returns a two-dimensional numpy array (matrix) with rows the number of permutations and columns the number of users
    """
    np.random.seed(seed)
    #Make an empty (zeros) matrix for the 100 random permutations
    random_perm_matrix = np.zeros((data_matrix.shape[0], num_permutations)) 
    random_perm = np.arange(data_matrix.shape[0])
    for i in range(num_permutations):
        #Make random permutations of the columns, and place them into the random permutation matrix
        np.random.shuffle(random_perm)
        random_perm_matrix[:,i] = random_perm
    print('The random matrix is\n',random_perm_matrix, "with shape:", random_perm_matrix.shape)

    #Make the signature matrix, which is initially filled with only inf-values
    signatureMatrix = np.full((random_perm_matrix.shape[1], data_matrix.shape[1]), np.inf)
    #Loop through the a set fraction of the total rows of the data we will use 
    for r in range(0, int(row_fraction*data_matrix.shape[0])): 
        random_perm_matrix_row = random_perm_matrix[r]
            
        #We first get the relevant row of the data, and then only loop over the non-zero elements! 
        data_matrix_row = data_matrix.getrow(r)
        data_col_indices = data_matrix_row.indices #indices of the non-zero elements
        
        #Loop over the non-zero elements
        for idx in data_col_indices:
            signatureMatrix[:,idx] = np.minimum(signatureMatrix[:,idx],random_perm_matrix_row)              

    print("\nSignature Matrix:\n", signatureMatrix)
    
    return np.array(signatureMatrix,dtype=np.int16)

def split_vector(signature, b, r):
    """
    Code splitting the signature matrix in b parts. The way we wrote the code, the following always holds:
    num_permutations = b*r. 

    Args:
        signature::[NumPy 2D array]
            The signature matrix created before. 
        b::[int]
            The number of bands we want to split the signature matrix in. 
        r::[int]
            The number of rows we want to split the signature matirx in. 
        
    Returns:
        subvecs::[NumPy 3D array]
            Returns a three-dimensional numpy array with axis 0 the number of bands, axis 1 the rows and axis 2 the number of users. 
    """
    subvecs = []
    for i in range(0, signature.shape[0],r):
        subvecs.append(signature[i : i+r])
    return np.array(subvecs,dtype=np.int32)

def generate_hash_function(size,total_users,seed):
    """
    Function to generate a random hash function. Based on a Linear Congruential Generator (LCG). 
    We use a different random seed for a and b to increase the randomness of the pseudo-RNG. 
    Args:
        size::[int]
            Amount of hash functions we want to produce 
        total_users::[int]
            The total number of users. 
        seed::[int]
            The NumPy seed used for the random functions.
        
    Returns:
        (Hash Function)::[tuple]
            Returns a hash function we can use to hash values to buckets. 
    """
    np.random.seed(seed)
    a = np.random.randint(1, 1000, size)
    np.random.seed(int(seed+1))
    b = np.random.randint(1, 1000, size)
    return lambda x: tuple((a * x + b) % total_users)

def hashing(hash_functions,usersTotal,b,split_S):

    #Hashing the bands to various buckets 
    hash_table = {}
    hash_counter = 0 
    candidate_pairs = []
    for u in range(usersTotal):  # Iterate over all users
        for band_idx in range(b):
            current_band = split_S[band_idx, :, u]

            #Apply the hash function to the band
            hashed_value = hash_functions[band_idx](tuple(current_band))

            if hashed_value not in hash_table:
                hash_table[hashed_value] = [(u, band_idx)]
            else:
                for stored_pair in hash_table[hashed_value]:
                    stored_u, stored_band_idx = stored_pair
                    if np.array_equal(current_band, split_S[stored_band_idx, :, stored_u]):
                        candidate_pairs.append((u, stored_u))
                        hash_counter += 1

                #Add the current pair to the hash table
                hash_table[hashed_value].append((u, band_idx))

    #Finally, we sort the candidate pairs such that we start off with the smallest u1 
    candidate_pairs = np.array(candidate_pairs)
    candidate_pairs[:,0], candidate_pairs[:,1] = candidate_pairs[:,1], candidate_pairs[:,0].copy()
    sorted_idx = np.argsort(candidate_pairs[:,0])
    candidate_pairs = candidate_pairs[sorted_idx]
    print('The number of candidate pairs found is:',np.shape(candidate_pairs)[0])

    return candidate_pairs

def jaccard_similarity(candidate_pairs,data):
    #In this function we use numpy to vectorize the Jaccard Similarity calculations for all candidate pairs
    #Using numpy drastically improves the efficiency of the code
    #The memory cost of converting sparse row matrices with datatype int64 costs a lot of RAM, so for that reason we convert the dtype to int8 first 
    users1 = data[:,candidate_pairs[:,0]].astype(np.int8).toarray()
    users2 = data[:,candidate_pairs[:,1]].astype(np.int8).toarray()

    num_pairs = len(candidate_pairs[:,0])
    simPairs = []
    simPairs_value = []
    with open('js.txt','w') as file:
        for i in range(num_pairs):
            one_gave_rating = np.logical_and(users1[:,i],users2[:,i]) #The logical AND operator represents the intersection
            both_gave_rating = np.logical_or(users1[:,i],users2[:,i]) #The logical OR operator represents the union 

            nom = np.sum(one_gave_rating)
            denom = np.sum(both_gave_rating)
            total = nom/denom
            if total > 0.5 and total not in simPairs_value:
                simPairs.append(candidate_pairs[i].tolist())
                simPairs_value.append(total)
                file.write(f"{candidate_pairs[i,0]}, {candidate_pairs[i,1]}\n")
                
    print('Accepted user pairs: \n',simPairs)
    print('Jaccard Similarity value: \n',simPairs_value)
    print('Number of similar user pairs:',len(simPairs_value))
    return simPairs,simPairs_value



def jaccardSim_results(simPairs_value):
    #This part is dedicated to plotting the similar users with a Jaccard Similarity above 0.5, with ascending order 
    plt.scatter(np.arange(len(simPairs_value)),np.sort(simPairs_value),s=5)
    plt.xlabel('Most similar pairs')
    plt.ylabel('Jaccard Similarity')
    plt.title('{} pairs with JS > 0.5'.format(len(simPairs_value)))
    plt.savefig('jc_similarity_result.png')

def apply_jaccardSim(data_path,seed):
    """
    Main function that calls all other functions to do the calculations of the Jaccard similarity
    """

    b = 35  #Number of bands we want to use to split the signature matrix in        
    r = 9   #The rows of values per band 
    m = 0.1 #fraction of rows that we want to pick a random permutation from. According to the book (top of page 88), the
            #resulting signature matrix should still be valid. This also increases speed of the calculation of the signature matrix 
            #by a factor 1/m, which helps a lot. 

    #Use 'seed' as argument for any function that uses RNG 

    data = loadData_js(data_path)
    sigMatrix = minhashing(data,m,int(b*r),seed) 
    split_sigMatrix = split_vector(sigMatrix,b,r)
    dim_bandedMatrix = np.shape(split_sigMatrix)
    print('The signature matrix split into bands has shape',dim_bandedMatrix)

    hash_functions = [generate_hash_function(r, dim_bandedMatrix[2],seed) for _ in range(b*10)]
    candidate_pairs = hashing(hash_functions,dim_bandedMatrix[2],b,split_sigMatrix)
    similar_pairs,similar_pairs_value = jaccard_similarity(candidate_pairs,data)    
    jaccardSim_results(similar_pairs_value)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#CS

def loadData_cs(data_path):
    """
    This function is dedicated to loading in the data, converting the data to a sparse col matrix (since the data is sparse) 
    

    Args:
        data_path::[str]
            The path to the data file, in this case user_movie_rating.npy
    Returns:
        sparse_colMatrix_full::[sparse matrix]
            Returns a sparse col matrix containing all the data, with movieIDs as rows and userIDs as columns 
    """
    # Load the full data
    data = np.load(data_path)
    
    # Create a sparse column matrix
    sparse_col_matrix = csc_matrix((data[:, 2], (data[:, 1], data[:, 0])))
    
    return sparse_col_matrix


def random_projections(data_matrix, num_permutations, seed):
    """
    This function creates a signature matrix using the sparse col matrix as input. To do so, the random projections theory described
    in the report is used. For more information on the general picture, please consult the report. 
    The size of the produced signature matrix can be influenced by choosing the number of permutations, num_permutations. 

    Args:
        data_matrix::[sparse matrix]
            Uses a sparse col matrix created using the data. 
        num_permutations::[int]
            Tells the code how many permutations to use for the signature matrix, directly corresponding to the number
            of rows the signature matrix will contain. 
        seed::[int]
            The NumPy seed used for the random functions.
        
    Returns:
        signatureMatrix::[NumPy 2D array]
            Returns a two-dimensional numpy array (matrix) with rows the number of permutations and columns the number of users
    """

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

def random_hash_functions(num_functions, num_buckets, seed):
    #create random hash functions
    
    hash_functions = []
    for _ in range(num_functions):
        a = np.random.randint(1, num_buckets)
        b = np.random.randint(0, num_buckets)
        hash_functions.append(lambda x: tuple((a * xi + b) % num_buckets for xi in x))
    return hash_functions

def apply_lsh_band(band_id, signature_matrix, hash_functions, rows_per_band, num_users):
    """
    Apply the banding technique to the signature matrix
    """
    
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

def apply_lsh_parallel(signature_matrix, hash_functions, num_bands, rows_per_band):
    """
    Apply the LSH technique to the signature matrix, using the random hash functions
    """
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

def find_candidate_pairs(bucket_lists, cosine_similarity_threshold):
    """
    This functions searches through all the buckets and finds candidate pairs
    """
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

def write_to_file(file_path, pairs):
    """
    This functions writes our final pairs to a txt file
    """
    with open(file_path, 'w') as f:  # Use 'a' mode to append to the file
        for pair in pairs:
            f.write(f"{pair[0]}, {pair[1]}\n")


def calculate_cosine_similarity(pair, sparse_col_matrix):
    """
    This file calculates the cosine similarity between two complete user-vectors 
    """
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

def CosineSim_results(final_pairs_cs):
    """
    This function makes a scatter plot of the sorted cosine similarity of the final pairs
    """
    # Scatter plot for most similar pairs
    x_data = np.arange(1, len(final_pairs_cs)+1)
    y_data = np.array([])
    for i in range(0, len(final_pairs_cs)):
        y_data = np.append(y_data, final_pairs_cs[i][2])
    y_data = np.sort(y_data)

    plt.scatter(x_data, y_data, s=5)
    plt.title('%i pairs with CS > 0.73' %len(x_data))
    plt.xlabel('Most similar pairs')
    plt.ylabel('Cosine Similarity')
    plt.savefig('cs_similarity_result.png')




def apply_cosineSim(data_path,seed):
    """
    Main function that calls all other functions to do the calculations of the cosine similarity
    """

    #Use 'seed' as argument for any function that uses RNG 

    data = loadData_cs(data_path)
    sparse_col_matrix = data

    # Create a signature matrix
    print("Create signature matrix...")
    begin = timeit.default_timer()
    signature_matrix = random_projections(data,100, seed) 
    end = timeit.default_timer() - begin
    print('Time to make signature matrix: ', end / 60, ' minutes')

    # Define LSH parameters
    num_hash_functions = 10
    num_buckets = 1000000
    num_bands = 2  # Experiment with the number of bands
    rows_per_band = 21  # Experiment with the number of rows per band
    #These values of (b=2, r=21) give us most pairs while remaining < 10 min on a 8 Gb RAM computer

    # Generate random hash functions
    hash_functions = random_hash_functions(num_hash_functions, num_buckets, seed)

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
    print("Number of candidate pairs:", len(candidate_pairs))

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
    write_to_file('cs.txt', final_pairs_cs)
    print('Final Pairs Execution Time: ', end / 60, ' minutes')
    print("Number of pairs with Cosine Similarity > 0.73:", len(final_pairs_cs), ", the final pairs are:\n", final_pairs_cs)
    CosineSim_results(final_pairs_cs)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#DCS

def calculate_discrete_cosine_similarity(pair, sparse_col_matrix):
    """
    This file calculates the discrete cosine similarity between two complete user-vectors 
    """
    u1, u2 = pair
    v1 = (sparse_col_matrix[:, u1].toarray()).flatten()
    v2 = (sparse_col_matrix[:, u2].toarray()).flatten()
    
    v1[v1 > 0] = 1
    v2[v2 > 0] = 1

    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)

    dot_product = np.dot(v1, v2)
    disc_cos_dist = np.arccos(dot_product / (mag_v1 * mag_v2))
    disc_cos_sim = 1 - disc_cos_dist / np.pi

    return u1, u2, disc_cos_sim
 
def DiscreteCosineSim_results(final_pairs_dcs):
    """
    This function makes a scatter plot of the sorted discrete cosine similarity of the final pairs
    """
    # Scatter plot for most similar pairs
    x_data = np.arange(1, len(final_pairs_dcs)+1)
    y_data = np.array([])
    for i in range(0, len(final_pairs_dcs)):
        y_data = np.append(y_data, final_pairs_dcs[i][2])
    y_data = np.sort(y_data)
 
    plt.scatter(x_data, y_data, s=5)
    plt.title('%i pairs with DCS > 0.73' %len(x_data))
    plt.xlabel('Most similar pairs')
    plt.ylabel('Discrete Cosine Similarity')
    plt.savefig('dcs_similarity_result.png')
 
def apply_discrete_cosineSim(data_path,seed):
    """
    Main function that calls all other functions to do the calculations of the discrete cosine similarity
    """
 
    #Use 'seed' as argument for any function that uses RNG 
 
    data = loadData_cs(data_path) #Same as for the cosine similarity, so we use the same function
    sparse_col_matrix = data
 
    # Create a signature matrix
    print("Create signature matrix...")
    begin = timeit.default_timer()
    signature_matrix = random_projections(data,100, seed) 
    end = timeit.default_timer() - begin
    print('Time to make signature matrix: ', end / 60, ' minutes')
 
    # Define LSH parameters
    num_hash_functions = 10
    num_buckets = 1000000
    num_bands = 2  # Experiment with the number of bands
    rows_per_band = 21  # Experiment with the number of rows per band
    #These values of (b=2, r=21) give us most pairs while remaining < 10 min on a 8 Gb RAM computer
 
    # Generate random hash functions
    hash_functions = random_hash_functions(num_hash_functions, num_buckets, seed)
 
    # Apply LSH in parallel
    begin = timeit.default_timer()
    bucket_lists = apply_lsh_parallel(signature_matrix, hash_functions, num_bands, rows_per_band)
    end = timeit.default_timer() - begin
    print('LSH Execution Time: ', end / 60, ' minutes')
 
    # Find candidate pairs
    begin = timeit.default_timer()
    candidate_pairs = find_candidate_pairs(bucket_lists,cosine_similarity_threshold=0.73) #same as for the cosine similarity, so re-use it
    end = timeit.default_timer() - begin
    print('Candidate Pairs Execution Time: ', end / 60, ' minutes')
    print("Number of candidate pairs:", len(candidate_pairs))
 
    # Convert candidate_pairs set to a list
    candidate_pairs = list(candidate_pairs)
 
    begin = timeit.default_timer()
    # Calculate discrete cosine similarity for candidate pairs
    final_pairs_dcs = []
    def process_batch(batch):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda p: calculate_discrete_cosine_similarity(p, sparse_col_matrix), batch))
        return [pair for pair in results if pair[2] > 0.73]
 
    batch_size = 5000
    for i in range(0, len(candidate_pairs), batch_size):
        batch = candidate_pairs[i:i + batch_size]
        final_pairs_dcs.extend(process_batch(batch))
    end = timeit.default_timer() - begin
    # Dump results to files
    write_to_file('dcs.txt', final_pairs_dcs)
    print('Final Pairs Execution Time: ', end / 60, ' minutes')
    print("Number of pairs with Discrete Cosine Similarity > 0.73:", len(final_pairs_dcs), ", the final pairs are:\n", final_pairs_dcs)
    DiscreteCosineSim_results(final_pairs_dcs)
    






def main(argv):
    data_file_path = ''
    random_seed = ''
    similarity_measure = ''
    try:
        opts, args = getopt.getopt(argv,"hd:s:m:",
                                   ["Data_file_path=","Random_seed=","Similarity_measure="])
    except getopt.GetoptError:
        print ('main.py -d <data file path> -s <random seed> -m <similarity measure>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('main.py -d <data file path> -s <random seed> -m <similarity measure>')
            sys.exit()
        elif opt in ("-d", "--Data_file_path"):
            data_file_path = arg
        elif opt in ("-s", "--Random_seed"):
            random_seed = arg
        elif opt in ("-m", "--Similarity_measure"):
            similarity_measure = arg
    print("Data File Path:", str(data_file_path))
    print("Random Seed:", int(random_seed))
    print("Similarity Measure:", str(similarity_measure))
 
    seed = int(random_seed)
    #Set the random seed of the code 
    np.random.seed(int(random_seed))
    if str(similarity_measure) == 'js':
        apply_jaccardSim(str(data_file_path),seed)
    elif str(similarity_measure) == 'cs':
        apply_cosineSim(str(data_file_path),seed)
    elif str(similarity_measure) == 'dcs':
        apply_discrete_cosineSim(str(data_file_path),seed)
    else:
        print("Please, choose one of these three options: [js,cs,dsc]")
 
if __name__ == "__main__":
    main(sys.argv[1:])







