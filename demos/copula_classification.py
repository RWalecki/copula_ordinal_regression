import copula_ordinal_regression as cor

# load the processed disfa database
X, y, _  = cor.load_disfa()

# select 3 outputs (AU1, AU2, AU12)
y = y[:,[0, 1, 6]]

# use 3000 samples for training and the rest for testing
X_tr, X_te = X[:3000],X[3000:]
y_tr, y_te = y[:3000],y[3000:]


clf = cor.COR(
        max_iter=5000,          # maximum number of iteration
        margins = 'sigmoid',    # marginal function. [ sigmoid, normcdf ]
        copula = 'frank',       # copula function.   [ frank, gumbel, indep ]
        optimizer = 'CG',       # scipy optimizer    [ CG, BFGS, TNC ... ]
        sparsity = 2,           # level of sparsity: use fully connected crf by setting this parameter to 0
        w_nodes = 0.1,          # balance potentials. set this parameter between 0 (only unary) and 1 (only binary)
        shared_copula = True,   # share the copula for all intensity levels
        verbose=1,              # verbose level
        )

# fit the model and apply predction 
clf.fit(X_tr,y_tr,debug=False)
y_hat = clf.predict(X_te)

# show results on test set
print(cor.metrics.ICC(y_te,y_hat))
print('avr. CORR:',cor.metrics.ICC(y_te,y_hat).mean())
print()
print(cor.metrics.CORR(y_te,y_hat))
print('avr. ICC:',cor.metrics.ICC(y_te,y_hat).mean())
