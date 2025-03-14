import itertools as it

def compute_log_likelihood(data, a, b, w,partition):
    log_likelihoods = [] # initialize list of LLHs
    Q=L
    conf = it.product((0,1), repeat=Q)
    all_conf=np.array(conf)
    for x in data: # From data (21770,784), takes one line each cycle, ---> len(x)=784
        Z_x = 0.0
        for z in all_conf:
            z = np.array(z)
            Hz = H(z)
            #print(f'for {x.shape}\n{z.shape}:{Hz.shape}')
            #E_xz = -np.sum(a * x) - np.sum(b * z) - np.sum(x * np.dot(W, z))
            E_xz = -np.dot(Hz,x) - np.dot(b,z)
            Z_x += np.exp(-E_xz)
            
        log_likelihoods.append(np.log(Z_x) - partition)# if Z_x != 0 and partition != 0 else '-Inf') ## first formula, not the second one.
    return np.mean(log_likelihoods)

def H(z): return a + np.dot(w, z)  # as in computations

def G(z): return np.prod(np.exp(b * z)) # as in pdf computations

def partition_function():
    lista_Gz_x_produttoria_h = np.array([])
    conf = it.product((0,1), repeat=L)
    all_conf=list(conf)
    for z in all_conf:
        Hz = H(z)
        #expHz = np.exp(Hz)
        #q = np.mean(1+expHz)
        produttoria_H = np.prod((1+np.exp(Hz)))
        Gzxproduttoria = G(z)*produttoria_H
        lista_Gz_x_produttoria_h = np.append(lista_Gz_x_produttoria_h, Gzxproduttoria)
    lnZ = np.log(np.sum(lista_Gz_x_produttoria_h))
    return lnZ

