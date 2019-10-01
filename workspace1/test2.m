function [errors] = test2(X,Y)
	echo off
%	Y = Y*2;
%	Y = Y-1;
%	X(:,size(X,2)+1) = 1;
	sz = size(X);
	num = sz(1);
	feat = sz(2);
	nv = [2,4,8,16];
	names = ["N=2 Train","N=4 Train","N=8 Train","N=16 Train","N = 2 Test","N = 4 Test","N = 8 Test","N = 16 Test"];
	etrainlr(1:4) = 0;
	etestlr(1:4) = 0;
	mask = make_xval_partition(num,4);
	trainx = X(mask ~= 1,:);
	trainy = Y(mask ~= 1,:);
	testx = X(mask == 1,:);
	testy = Y(mask == 1,:);
	sz = size(trainx)

	e0(1:8) = 0;
	its = 10;

	for it = 1:its;
	for i = 1:4
		p = make_xval_partition(sz(1),nv(i));

		etrainlr(i) = logistic_xval_error(trainx, trainy, p);
		etestlr(i) = logistic_xval_test(trainx, trainy, testx, testy, p);
	end
	errors = [etrainlr,etestlr];
	e0 = e0 + errors;
	end

	e0 = e0/its;
	


	figure;
	bar(e0);
	set(gca, 'xticklabel', names);
	title('Train Error & Test Error ,Original Data');

end
