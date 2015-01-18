function dispimages(I,data,data_red)
%Displays image as well as the data point corresponding to that plot
for i=1:length(I)
    fprintf('image %d\n',i);
    im=reshape(data(I(i),:),64,64); 
    figure(2);
    imshow(im,[]);
    figure(1);
    plot(data_red(:,1),data_red(:,2),'.');
    hold on;
    plot(data_red(I(i),1),data_red(I(i),2),'ko','MarkerFaceColor','r');
    hold off;
    pause;
end
end