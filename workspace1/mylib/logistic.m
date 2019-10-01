function precision = logistic(X_train, Y_train, X_test, Y_test)

model = train(Y_train, sparse(X_train), ['-s 0', 'col']);
[predicted_label] = predict(Y_test, sparse(X_test), model, ['-q', 'col']);

precision = 1 - sum(predicted_label~=Y_test) / length(Y_test);

end

