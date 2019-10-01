%[score_train, wtf, numpc] = pca_getpc(Y, [1]);
%Yp = score_train(:,1:1);
%size(Yp)
%corr(Yp, Y)

[score_train, score_test, numpc, coeff] = pca_getpc(X_train, X_test);
%mask0 = (Y_test == 1);
%mask1 = (Y_test == 2);
%x1 = s2_test(:,1:1);
%x2 = s2_test(:,2:2);
%x10 = x1(mask0);
%x11 = x1(mask1);
%x20 = x2(mask0);
%x21 = x2(mask1);
%plot(x10,x20,'o',x11,x21,'+')
%legend('0','1')
%title('Plot of 0 and 1 digits from top 2 PCA dimensions')
%xlabel('PC1')
%ylabel('PC2')

pc = [100,150,200];

for i = 1:3
	Xptr = score_train(:,1:pc(i));
	Xpte = score_test(:,1:pc(i));
    precision_ori_km(i) = k_means(Xpte, Y_test, Xpte, Y_test, 25);
end
precision_ori_km




