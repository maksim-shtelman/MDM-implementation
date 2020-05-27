import numpy as np

class MDM():
    """
        Implementation of MDM(Mitchell-Demyanov-Malozemov) algorithm
        for binary classification
    """

    def __init__(self, max_iter=100, kernel_type='linear', C=1.0, epsilon=0.00001):
        self.kernels = {
            'linear' : self.kernel_linear,
            'quadratic' : self.kernel_quadratic
        }
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.C = C
        self.epsilon = epsilon

    def fit(self, X, y):
        # Initializing
        n_samples, n_features = X.shape[0], X.shape[1]
        alpha = np.zeros((n_samples))
        kernel = self.kernels[self.kernel_type]
        count = 0
        alpha[0] = self.C
        for j in range(1, n_samples):
            if(y[j] != y[0]):
                alpha[j] = self.C
                break
        d =  np.zeros(n_samples)
        self.w = self.calc_w(alpha, y, X)
        while True:
            count += 1
            for j in range(n_samples):
                d[j] = np.dot(np.transpose(self.w),X[j])

            #    i1+
            dmax = min(d)
            for j in range(n_samples):
                 if((alpha[j]>0)and(d[j]>dmax)and(y[j]==1)):
                    dmax = d[j]
                    iplus1 = j
            #   i1-
            dmin = max(d)
            for j in range(n_samples):
                 if((alpha[j]>0)and(d[j]<dmin)and(y[j]==-1)):
                    dmin = d[j]
                    iminus1 = j
            #   i2+
            dmin = max(d)
            for j in range(n_samples):
                 if((d[j]<dmin)and(y[j]==1)and(j!=iplus1)and(j!=iminus1)and(alpha[j] < self.C)):
                    dmin = d[j]
                    iplus2 = j
            #   i2-
            dmax = min(d)
            for j in range(n_samples):
                 if((d[j]>dmax)and(y[j]==-1)and(j!=iplus1)and(j!=iminus1)and(alpha[j] < self.C)):
                    dmax = d[j]
                    iminus2 = j


            ## choosing i1, i2
            dplus = d[iplus2] - d[iplus1]
            dminus = d[iminus2] - d[iminus1]

            if(dplus>dminus):
                i1 = iplus1
                i2 = iplus2
            else:
                i1 = iminus1
                i2 = iminus2

            Z = X[i2] - X[i1]
            delta = y[i2]*np.dot(self.w,Z)/(np.linalg.norm(Z))**2 ## delta_i1 (-delta_i2)
            delta = min(-delta, self.C - alpha[i2], alpha[i1])
            w_prev = self.w

            ## updating alpha
            alpha[i1] = alpha[i1] - delta
            alpha[i2] = alpha[i2] + delta
            ## /изменения

            # updating W
            self.w = self.calc_w(alpha, y, X)
            self.b = self.calc_b(X, y, self.w)

            # checking convergence
            diff = np.abs(np.linalg.norm(w_prev)**2-np.linalg.norm(self.w)**2)
            if diff < self.epsilon*np.linalg.norm(w_prev)**2:
                print(count)
                break


            if count >= self.max_iter:
                print("Iteration number exceeded the max of %d iterations" % (self.max_iter))
                print(diff)
                return
        # Computing final parameters
        self.b = self.calc_b(X, y, self.w)
        if self.kernel_type == 'linear':
            self.w = self.calc_w(alpha, y, X)
        # Getting support vectors
        alpha_idx = np.where(alpha > 0)[0]
        support_vectors = X[alpha_idx, :]
        print('Number of iterations: %i' % count)
        return support_vectors, count
    def predict(self, X):
        return self.h(X, self.w, self.b)
    def calc_b(self, X, y, w):
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)
    def calc_w(self, alpha, y, X):
        return np.dot(X.T, np.multiply(alpha,y))
    # Prediction
    def project(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.sign(self.project(X))

    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)
    def kernel_quadratic(self, x1, x2):
        return (np.dot(x1, x2.T) ** 2)
    # accuracy scoring for a model
    def score(self, X, y):
        y_pred = self.predict(X)
        sum = 0
        for j in range(y.shape[0]):
            if(y[j]==y_pred[j]):
                sum += 1
        return sum/y_test.shape[0]
    # f1 scoring
    def f1(self, X, y):
        y_pred = self.predict(X)
        TP = 0
        FN = 0
        FP = 0
        for j in range(y.shape[0]):
            if(y[j]==y_pred[j]==1):
                TP += 1
            if((y[j]!=y_pred[j])and(y_pred[j]==1)):
                FP += 1
            if((y[j]!=y_pred[j])and(y_pred[j]==-1)):
                FP += 1
        if(TP==FN==FP==0):
            return 0
        Precision = TP/(TP+FP)
        if(TP == 0):
            return 0
        else:
            Recall = TP/(TP+FN)
        return 2*(Precision*Recall)/(Precision+Recall)
