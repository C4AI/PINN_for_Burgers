function deriv = finiteDifferencesFunc(X,method,dir)

n = length(X);

[centeredStencilLHS, decenteredStencilLHS, decenteredStencilRHS, centeredStencilRHS, centeredStencilRHS2, decenteredStencilRHS2] = finiteDifferenceCoefficients(method);

LHS = makeEachMatrix(centeredStencilLHS,decenteredStencilLHS,n);
RHS = makeEachMatrix(centeredStencilRHS,decenteredStencilRHS,n);

LHS = sparse(LHS);
RHS = sparse(RHS);

% If the second derivative is defined for the chosen method, compute its RHS
if ~isempty(centeredStencilRHS2)
    RHS2 = makeEachMatrix(centeredStencilRHS2,decenteredStencilRHS2,n);
    RHS2 = sparse(RHS2);
end

% Apply metrics
dXdEta = LHS\(RHS*X');
LHS = bsxfun(@times,LHS,dXdEta');

if dir == 1 % For X direction
    
    deriv{1} = @(U)(LHS\(RHS*U));
    if ~isempty(centeredStencilRHS2)
        deriv{2} = @(U)(LHS.^2\(RHS2*U));
    else
        % If there are no coefficients for the second derivative, apply first derivative twice
        deriv{2} = @(U)(deriv{1}(deriv{1}(U)));
    end
else % For Y direction
    
    % Transpose LHS and RHS
    LHS = LHS';
    RHS = RHS';
    
    deriv{1} = @(U)((U*RHS)/LHS);
    if ~isempty(centeredStencilRHS2)
        RHS2 = RHS2';
        deriv{2} = @(U)((U*RHS2)/LHS.^2);
    else
        % If there are no coefficients for the second derivative, apply first derivative twice
        deriv{2} = @(U)(deriv{1}(deriv{1}(U)));
    end
end


end
