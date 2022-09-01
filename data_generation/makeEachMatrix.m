function M = makeEachMatrix(C,D,n)
  % This builds either LHS or RHS matrices based on the finite differences coefficients
  M = zeros(n,n);
  lc = length(C);
  [nd,ld] = size(D);

  M(1:nd,1:ld) = D;
  if C(1) == 0 % C(1)==0 for the first derivative's RHS
      M(end:-1:end-nd+1,end:-1:end-ld+1) = -D;
  else
      M(end:-1:end-nd+1,end:-1:end-ld+1) = D;
  end

  if C(1) == 0
      C = [-C(end:-1:2) C];
  else
      C = [C(end:-1:2) C];
  end

  for i = nd+1:n-nd
      M(i,i-lc+1:i+lc-1) = C;
  end

end