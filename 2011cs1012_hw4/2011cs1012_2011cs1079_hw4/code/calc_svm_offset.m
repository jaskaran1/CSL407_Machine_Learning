function w_0=calc_svm_offset(alpha,Y,K,C)
%Calculates w_0,the offset of the hyperplane W'x+w_0=0
indsv=find(alpha>0 & alpha<C);
Ysv=Y(indsv,:);%Ysv is a column vector
Ksv=K(indsv,:);%Nsv X n matrix
Nsv=length(indsv);
w_0=sum(Ysv-Ksv*(Y.*alpha))/Nsv;
end
