#Import all the neccessary libraries
import numpy as np
from scipy.sparse import csr_matrix,csc_matrix
import timeit

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

#This part is dedicated to plotting the similar users with a Jaccard Similarity above 0.5, with ascending order 

def jaccardSim_results(simPairs_value):
    import matplotlib.pyplot as plt

    plt.scatter(np.arange(len(simPairs_value)),np.sort(simPairs_value),s=5)
    plt.xlabel('Most similar pairs')
    plt.ylabel('Jaccard Similarity')
    plt.title('{} pairs with JS > 0.5'.format(len(simPairs_value)))
    plt.savefig('jc_similarity_result.png')

def apply_jaccardSim(data_path,seed):

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

#apply_jaccardSim('user_movie_rating.npy',1702)

import sys, getopt

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
        print('we dont have that yet')
    elif str(similarity_measure) == 'dcs':
        print('we dont have that yet')
    else:
        print("Please, choose one of these three options: [js,cs,dsc]")
 
if __name__ == "__main__":
    main(sys.argv[1:])
