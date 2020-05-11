function [muhat,s2hat] = tstfitendmodel(mu,s2,npts,obsvarstddev)

	if nargin<3
		npts = 1000;
	end
	if nargin<4
		obsvarstddev=25;
	end

	vs = rand(npts,1)*obsvarstddev;
	vals = randn(npts,1).*sqrt(vs+s2)+mu;
	[muhat,s2hat] = fitendmodel(vals,vs);
