from sklearn.ensemble import RandomForestClassifier


clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

#training on just a 1000 pixels to test out model
clf.fit(X_train[:1000], y_train[:1000])