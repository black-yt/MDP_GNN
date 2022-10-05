clc;clear;
edges=xlsread('edges.xlsx','1');
N=19717;
A=zeros(19717);
[nn,~]=size(edges);
for ii=1:1:nn
    A(edges(ii,1)+1,edges(ii,2)+1)=1;
    A(edges(ii,2)+1,edges(ii,1)+1)=1;
end

A=A+eye(N:N);

[u,v]=eig(A);

choose=rand(N);
choose(choose>=0.5)=1;
choose(choose<0.5)=0;
key=diag(diag(choose));
v_1=v.*key;
out_1=u*v_1/u;

out_1=real(out_1);
out_1=single(out_1);
out_2=A-out_1;
out_2=single(out_2);

[i,j]=find(out_1>0.5);
m1=[i,j];
[i,j]=find(out_2>0.5);
m2=[i,j];