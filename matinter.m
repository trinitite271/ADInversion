function [] = matinter()
addpath('/home/user/zhangchang/matlab/WT_P2/WT_P/core')
load('matinverarg.mat');
myvars = who;
for i=1:length(myvars)
    eval([char(myvars(i)) '=double(' char(myvars(i)) ');']);
end
damp=damp_circle(vmin,nzbc,nxbc,nbc,dx,isfs,pad_top);
temp=1-damp*dt;
[s,~] = ricker(f,dt,nt);
save('matinvertemp.mat','temp','s');
end
