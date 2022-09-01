function [U1,V1] = timestepper(U0,V0,method,dt,nu,dx,dy)

    switch method
        case 'Euler'
            [Ut,Vt] = burgereq(U0,V0,nu,dx,dy);
            U1 = U0 + dt*Ut;
            V1 = V0 + dt*Vt;

        case 'RK4'
            [Ut1,Vt1] = burgereq(U0,V0,nu,dx,dy);
            [Ut2,Vt2] = burgereq(U0+dt/2*Ut1,V0+dt/2*Vt1,nu,dx,dy);
            [Ut3,Vt3] = burgereq(U0+dt/2*Ut2,V0+dt/2*Vt2,nu,dx,dy);
            [Ut4,Vt4] = burgereq(U0+dt*Ut3,V0+dt*Vt3,nu,dx,dy);

            U1 = U0 + dt/6*(Ut1 + 2*Ut2 + 2*Ut3 + Ut4);
            V1 = V0 + dt/6*(Vt1 + 2*Vt2 + 2*Vt3 + Vt4);
    end

end