mypath = "GlassPCA/glass_data.txt"
def PCA_analysis(path):
    import numpy as np
    iris_data = np.loadtxt(path, comments="%")
    # x is a matrix with 50 rows and 4 columns
    x = iris_data
    n_feat = x.shape[1]
    n_obs = x.shape[0]
    print(f"Number of features: {n_feat} and number of observations: {n_obs}")

    mn = np.mean(x, axis=0)
    data = x - mn

    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    spread = maxs - mins
    data = data / spread

    print(f"Answer: Amount of Sodium {data[0][1]:.2f}")
    c_x = np.cov(data.T)

    print(f"Answer: Covariance matrix at (0, 0): {c_x[0][0]:.3f}")


    values, vectors = np.linalg.eig(c_x) # Here c_x is your covariance matrix.
    v_norm = values / values.sum() * 100
    plt.plot(v_norm)
    plt.xlabel("Principal component")
    plt.ylabel("Percent explained variance")
    plt.ylim([0, 100])

    plt.show()

    #project data
    pc_proj = vectors.T.dot(data.T)

    return v_norm, abs(pc_proj).max()

PCA_analysis(path=mypath)
