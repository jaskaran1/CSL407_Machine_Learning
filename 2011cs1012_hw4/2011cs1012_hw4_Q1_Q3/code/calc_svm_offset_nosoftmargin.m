function w_0=calc_svm_offset_nosoftmargin(alpha,Y,K)
%Calculates w_0,the offset of the hyperplane W'x+w_0=0
indsv=find(alpha>0);
Ysv=Y(indsv,:);%Ysv is a column vector
Ksv=K(indsv,:);%Nsv X n matrix
Nsv=length(indsv);
w_0=sum(Ysv-Ksv*(Y.*alpha))/Nsv;
end
