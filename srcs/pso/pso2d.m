global  W1 W2 W3 W4 B1 B2 B3 B4 x_means x_stds y_means y_stds normalized_o all_fitness

str='val.xlsx';
load('weight.mat');
load('means_stds.mat');

dataset=table2array(readtable(str));
x=dataset(:,1:6);
y=dataset(:,7:18);
val_index=3;
o=y(val_index,:);
gt=x(val_index,:);
gtz=(gt-x_means)./x_stds;
figure(1)
scatter(gtz(5),gtz(6),2000,'+');
xlabel('Length');
ylabel('Diameter');
grid on
hold on
normalized_o = normalized_y(o);
tic
gto=forward(gtz);
toc

predict_out=gto.*y_stds+y_means;
pop=100;
dim=2;
constraint_dim=4;
ub=[2,4];
lb=[-2,-2];
vmax=[1,1];
vmin=[-1,-1];
a=0.4;
b=0.4;
vmax=a*vmax;
vmin=b*vmin;
maxIter=50;
all_fitness=ones(pop,maxIter);
global position bp
position=zeros(maxIter,pop,2);
bp=zeros(maxIter,2);
fobj=@(X)fun(X);
[Best_Pos,Best_fitness,IterCurve]=pso(pop,dim,constraint_dim,ub,lb,...
    fobj,vmax,vmin,maxIter);
figure
plot(IterCurve,'r','linewidth',2);
grid on;
reverse_x(Best_Pos)
no=forward(Best_Pos);
x=squeeze(position(:,:,1));
y=squeeze(position(:,:,2));
x=x*x_stds(5)+x_means(5);
y=y*x_stds(6)+x_means(6);
all_fitness=log(all_fitness);
xp=position(:,:,1);
yp=position(:,:,2);
function o=errors(x1,x2)
weight_matrix=[
    0.2 ...
    0.2 ...
    0.1 ...
    0.01 ...
    0.3 ...
    0.3 ...
    0.3 ...
    0.3 ...
    0.4 ...
    0.4 ...
    0.4 ...
    0.4];
error=(x1-x2).^2;
error=error';
o=weight_matrix*error;
end

function [Best_Pos,Best_fitness,IterCurve]=pso(pop,dim,cdim,ub,lb,...
    fobj,vmax,vmin,maxIter)

global position bp all_fitness
best_position_iter=zeros(maxIter,2);
IterCurve=ones(1,maxIter);
c1=1.4;
c2=1.4;
wmax=0.9;
wmin=0.1;

V=initialization(pop,vmax,vmin,dim);
X=initialization(pop,ub,lb,dim);
Constrain_variables=zeros(pop,cdim);
Com=[Constrain_variables X];
for i=1:pop
    p1=rand(1,1);
    Com(i,2)=get_p(p1);
    n=round(rand(1,1)*3)+7;
    n1=round(p1*6+1);
    Com(i,4)=amount_normalize(n);
    Com(i,3)=d_calculate(n1,Com(i,6),n);
    Com(i,1)=ar_modulation(Com(i,5),Com(i,6));
end
fitness=zeros(1,pop);
for i=1:pop
    fitness(i)=fobj(Com(i,:));
end
pBest=Com(:,5:6);
pBestFitness=fitness;
[~,index]=min(fitness);
gBestFitness=fitness(index);
gBest=pBest(index,:);
GBEST=Com(index,:);
Xnew=pBest;
fitnessNew=fitness;
for t=1:maxIter
    tic
    w=wmax-(wmax-wmin)*t/maxIter;
    for i=1:pop
        local_best_p=1;
        local_best_n=1;
        local_best_fitness=inf;
        r1=rand(1,dim);
        r2=rand(1,dim);
        V(i,:)=w.*V(i,:)+c1.*r1.*(pBest(i,:)-X(i,:))+c2.*r2.*(gBest-X(i,:));
        V(i,:)=BoundaryCheck(V(i,:),vmax,vmin,dim);
        Xnew(i,:)=X(i,:)+V(i,:);
        Xnew(i,:)=BoundaryCheck(Xnew(i,:),ub,lb,dim);
        Com=[Constrain_variables Xnew];
        Com(i,1)=ar_modulation(Com(i,5),Com(i,6));
        for j=1:1:7
            Com(i,2)=pitch(j);
            for n=6:1:25
                Com(i,4)=amount_normalize(n);
                Com(i,3)=d_calculate(j,Com(i,6),n);
                fitnessNew(i)=fobj(Com(i,:));
                if fitnessNew(i)<local_best_fitness
                    local_best_fitness=fitnessNew(i);
                    local_best_p=j;
                    local_best_n=n;
                end
            end
        end
        Com(i,2)=pitch(local_best_p);
        Com(i,4)=amount_normalize(local_best_n);
        Com(i,3)=d_calculate(local_best_p,Com(i,6),local_best_n);
        fitnessNew(i)=fobj(Com(i,:));
        if fitnessNew(i)<pBestFitness(i)
            pBest(i,:)=Xnew(i,:);
            pBestFitness(i)=fitnessNew(i);
        end
        if fitnessNew(i)<gBestFitness
            gBestFitness=fitnessNew(i);
            gBest=Xnew(i,:);
            GBEST=Com(i,:);
        end
        all_fitness(i,t)=fitnessNew(i);
    end
    toc
    X=Xnew;
    Best_Pos=GBEST;
    bp(t,:)=gBest;
    scatter(Xnew(:,1),Xnew(:,2),10,'*');
    position(t,:,1)=Xnew(:,1);
    position(t,:,2)=Xnew(:,2);
    hold on
    Best_fitness=gBestFitness;
    IterCurve(t)=gBestFitness;
    disp(t)
    %Best_fitness
    best_position_iter(t,:)=gBest;
end
plot(best_position_iter(:,1),best_position_iter(:,2),LineWidth=2);
end
function p=get_p(a)
pz=[1.27391031742826	0.678164111414312...
    0.201760257204782 -0.0365382252007945...
    -1.10868828542427	-1.34698676782984	-1.70443449143821];
a=a*6;
a=round(a+1);
p=pz(a);
end
function p=pitch(a)
pz=[1.27391031742826	0.678164111414312...
    0.201760257204782 -0.0365382252007945...
    -1.10868828542427	-1.34698676782984	-1.70443449143821];
p=pz(a);
end
function x_original = reverse_x(in)
global x_means x_stds
x_original = (in.*x_stds)+x_means;
end
function y_normal = normalized_y(in)
global y_stds y_means
y_normal = (in-y_means)./y_stds;
end
function n=amount_normalize(in)
n_means=11.631353;
n_stds=3.622755;
n=(in-n_means)/n_stds;
end
function out = ar_modulation(L_z,W_z)
L_means=67.9663;
A_means=4.794;
W_means=14.307348;
L_stds=19.626038;
A_stds=1.313446;
W_stds=2.721691;
out=(((L_z*L_stds+L_means)/(W_z*W_stds+W_means))-A_means)/A_stds;
end
function d_normalized = d_calculate(pitch_index,w_z,n)
d_means=19.50966;
d_stds=10.7106;
w_means=14.307348;
w_stds=2.721691;
p=[407.16 376.31 351.64 339.3 283.78 271.44 255.93];
pitch=p(pitch_index);
d_original=(pitch-n*(w_z*w_stds+w_means))/(n-1);
d_normalized=(d_original-d_means)/d_stds;
end

function [X]=initialization(pop,u,l,dim)
X=zeros(pop,dim);
for i=1:pop
    for j=1:dim
        X(i,j)=(u(j)-l(j))*rand()+l(j);
    end
end
end
function y = act(x)
alpha = 0.1;
y = max(alpha*x, x);
end
function fitness=fun(x)
global W1 W2 W3 W4 B1 B2 B3 B4 normalized_o
layer1=act(x*W1+B1);
layer2=act(layer1*W2+B2);
layer3=act(layer2*W3+B3);
net_o=layer3*W4+B4;
fitness=errors(normalized_o,net_o);
x_ori=reverse_x(x);
if(x_ori(3)<x_ori(6)/2+0.2)
    fitness=inf;
end
end
function [X]=BoundaryCheck(X,ub,lb,dim)
for i=1:dim
    if X(i)>ub(i)
        X(i)=ub(i);
    end
    if X(i)<lb(i)
        X(i)=lb(i);
    end
end
end
