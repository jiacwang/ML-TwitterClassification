function [Xtrains] = pre_processing(X_train,training)

	X_train = [sum(X_train,2),X_train];

	if (training == 1)

		numpc = 20;

		[U,S,V] = svds(X_train,numpc);

%		Xtrains = [full(sum(X_train,2)), U];
		Xtrains = U;
		Xtrains = Xtrains*S;

		save svd.mat S V

	else
		
		load svd.mat

%		Xtrains = [full(sum(X_train,2)), X_train * V ./ diag(S)'];
		Xtrains = X_train * V; %./ diag(S)';


	end

end


