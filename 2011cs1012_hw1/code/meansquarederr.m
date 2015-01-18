function err=meansquarederr(T,Tdash)
n=length(T);
err=sum((Tdash-T).^2)/n;
end