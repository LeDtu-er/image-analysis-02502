import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
import glob

def pca_analysis(path, dtype = "txt", skl = True):
    """pca_analysis

    Args:
        path (str): path to either folder or the specific data file
        dtype (str, optional): whether the data is given in txt or images. Defaults to "txt".
        skl (bool, optional): whether to do everything manually or use sklearn. 
                              note: if we have way more features than examples it will automatically 
                              use sklearn. otherwise you computer will explode. Defaults to True.

    Raises:
        SyntaxError: provide either txt or image

    Returns:
        eig_val: eigen values
        eig_vect: eigen vectors
        var_expl: variance explained (not cummulitative)
        pc_proj: data projected into principal component space
    """
    
    if dtype == "txt":
        data = np.loadtxt(path, comments="%")
    elif dtype == "image":
        images = glob.glob(path + "*")
        w, h = plt.imread(images[0]).shape
        
        data = np.zeros((len(images), w * h))
        
        for idx, image in enumerate(images):
            img_data = plt.imread(image)
            data[idx] = img_data.flatten()   
    else:
        raise SyntaxError("You need to specify how the data should be read")

    if not skl or data.shape[0] >= data.shape[1]:
        # centralize
        data = (data - data.mean(axis = 0))
        
        c_x = 1/(len(data[:,0])-1)*(data.T @ data)
        eig_values, eig_vectors = np.linalg.eig(c_x) # Here c_x is your covariance matrix.
        
        v_explained = eig_values / eig_values.sum() * 100
        
        # project data onto pca vectors
        pc_proj = eig_vectors.T.dot(data.T)
    else:
        pca = decomposition.PCA()
        pca.fit(data)
        pca.fit_transform()
        eig_values = pca.explained_variance_
        v_explained = pca.explained_variance_ratio_
        eig_vectors = pca.components_
        eig_vectors = eig_vectors.T
        
        pc_proj = eig_vectors.T.dot(data.T)
    
    # MIGHT fail... for some reason the f22 set doesn't give the right answer
    return eig_values, eig_vectors, v_explained, pc_proj