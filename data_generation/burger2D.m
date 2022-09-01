%% Numeric parameters

if ~exist('runningBatch','var')
    x0 = 0;
    x1 = 1;
    nx = 101;

    y0 = 0;
    y1 = 1;
    ny = 101;

    tf = 0.5;
    dt = 1e-5;

    nu = 0.01/pi;

    finiteDifferencesMethod = 'EX2';
    timeSteppingMethod = 'RK4';
    meshType = 'uniform'; % It's best to use 'uniform' for regular finite differences and 'cheb' for chebyshev differentiation

    saveSteps = 100;
    plotSteps = 250;
    saveImages = true;
end

%% Generate mesh

switch meshType
    case 'uniform'
        X = linspace(x0,x1,nx);
        Y = linspace(y0,y1,ny);
    case 'cheb'
        X = cos(linspace(0,pi,nx));
        X = (1-X)/2*(x1-x0)+x0;
        Y = cos(linspace(0,pi,ny));
        Y = (1-Y)/2*(y1-y0)+y0;
end

T = 0:dt*saveSteps:tf;
nt = length(T);

%% Generate initial conditions
xhat = (X-x0)/(x1-x0);
yhat = (Y'-y0)/(y1-y0);

U0 = bsxfun(@times,sin(2*pi*xhat),sin(2*pi*yhat));
V0 = bsxfun(@times,sin(pi*xhat),sin(pi*yhat));

U0([1 end],:) = 0;
U0(:,[1 end]) = 0;
V0([1 end],:) = 0;
V0(:,[1 end]) = 0;

%% Initialize solution
U = nan(ny,nx,nt);
V = nan(ny,nx,nt);

U(:,:,1) = U0;
V(:,:,1) = V0;

%% Generate matrices for finite differences
dx = finiteDifferencesFunc(X,finiteDifferencesMethod,2);
dy = finiteDifferencesFunc(Y,finiteDifferencesMethod,1);

%% Time stepping

Uold = U0;
Vold = V0;
nSave = 1;
nImage = 1;

for i = 1:(nt-1)*saveSteps
    
    disp(i)
    
    [Unew, Vnew] = timestepper(Uold,Vold,timeSteppingMethod,dt,nu,dx,dy);
    
    if mod(i,saveSteps)==0
        nSave = nSave + 1;
        
        U(:,:,nSave) = Unew;
        V(:,:,nSave) = Vnew;
    end
    
    if mod(i,plotSteps)==0
        step = 4;
        surf(X(1:step:end),Y(1:step:end),Unew(1:step:end,1:step:end))
        view(-15,25)
        drawnow
        if saveImages
            saveas(gcf,['images/' num2str(nImage,'%04d') '.png']);
        end
        nImage = nImage + 1;
    end
    
    Uold = Unew;
    Vold = Vnew;
end