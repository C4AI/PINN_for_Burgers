function [Ut,Vt] = burgereq(U,V,nu,dx,dy)

    Ut = nu*(dx{2}(U) + dy{2}(U)) - U.*dx{1}(U) - V.*dy{1}(U);
    Vt = nu*(dx{2}(V) + dy{2}(V)) - U.*dx{1}(V) - V.*dy{1}(V);

    Ut([1 end],:) = 0;
    Ut(:,[1 end]) = 0;

    Vt([1 end],:) = 0;
    Vt(:,[1 end]) = 0;

end