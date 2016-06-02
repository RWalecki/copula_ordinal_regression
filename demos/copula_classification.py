import copula_ordinal_regression as cor

# load the processed disfa database
X, y, _  = cor.load_disfa()

# select the first 3 action units (AU1,AU2,AU4)
y = y[:,[0,1,2]]

# use 3000 samples for training and the rest for testing
X_tr, X_te = X[:3000],X[3000:]
y_tr, y_te = y[:3000],y[3000:]


clf = cor.COR(max_iter=1000, verbose=1)

# fit the model and apply predction 
clf.fit(X_tr,y_tr)
y_hat = clf.predict(X_te)

# print resutls on test set
print cor.metrics.ICC(y_te,y_hat)
print 'avr. CORR:',cor.metrics.ICC(y_te,y_hat).mean()
print 
print cor.metrics.CORR(y_te,y_hat)
print 'avr. ICC:',cor.metrics.ICC(y_te,y_hat).mean()
