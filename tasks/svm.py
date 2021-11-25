import numpy as np
import icecream


class linearSVM:

    def __init__(self):
        self.W = None

    def train(self,  X, y, lr=1e-3, reg=1e-5, num_iter=100, verbose=False):
        train_num, dim = X.shape
        num_classes = (np.max(y) + 1)
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        loss_history = []
        for it in range(num_iter):
            loss, grad = self.loss(X, y, reg)
            loss_history.append(loss)
            self.W += -lr * grad
            if verbose and it % 5 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iter, loss))

        return loss_history

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i, each in enumerate(X.dot(self.W)):
            pred = each.tolist().index(max(each))
            y_pred[i] = pred
        return y_pred

    def loss(self, X, y, reg):
        return self.svm_loss_vectorized(self.W, X, y, reg)

    def svm_loss_vectorized(self, W, X, y, reg):
        """
        Structured SVM loss function, vectorized implementation.

        Inputs and outputs are the same as svm_loss_naive.
        """
        loss = 0.0
        dW = np.zeros(W.shape)  # initialize the gradient as zero
        num_train = X.shape[0]


        #############################################################################
        # Implement a vectorized version of the structured SVM loss, storing the    #
        # result in loss.                                                           #
        #############################################################################
        scores = X.dot(W)
        yi_scores = scores[np.arange(num_train), y]  # http://stackoverflow.com/a/23435843/459241
        margins = np.maximum(0, scores - np.matrix(yi_scores).T + 1)
        margins[np.arange(num_train), y] = 0
        loss = np.mean(np.sum(margins, axis=1))
        loss += 0.5 * reg * np.sum(W * W)

        #############################################################################
        # Implement a vectorized version of the gradient for the structured SVM     #
        # loss, storing the result in dW.                                           #
        #                                                                           #
        # Hint: Instead of computing the gradient from scratch, it may be easier    #
        # to reuse some of the intermediate values that you used to compute the     #
        # loss.                                                                     #
        #############################################################################
        binary = margins
        binary[margins > 0] = 1
        row_sum = np.sum(binary, axis=1)
        binary[np.arange(num_train), y] = -row_sum.T
        dW = np.dot(X.T, binary)

        # Average
        dW /= num_train

        # Regularize
        dW += reg * W

        return loss, dW



