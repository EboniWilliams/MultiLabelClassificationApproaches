function [Prior,PriorN,Cond,CondN,]=MLKNN_train(train_data,train_target,Num,Smooth)


    [num_class,num_training]=size(train_target);

%Computing distance between training instances
        mat1=concur(sum(train_data.^2,2),num_training);
        mat2=mat1';
        dist_matrix=mat1+mat2-2*train_data*train_data';
        dist_matrix=sqrt(dist_matrix);
        for i=1:num_training
            dist_matrix(i,i)=realmax;
        end

        %Computing Prior and PriorN
        for i=1:num_class
            temp_Ci=sum(train_target(i,:)==ones(1,num_training));
            Prior(i,1)=(Smooth+temp_Ci)/(Smooth*2+num_training);
            PriorN(i,1)=1-Prior(i,1);
        end

        %Computing Cond and CondN
        Neighbors=cell(num_training,1); %Neighbors{i,1} stores the Num neighbors of the ith training instance
        for i=1:num_training
            [temp,index]=sort(dist_matrix(i,:));
            Neighbors{i,1}=index(1:Num);
        end

        temp_Ci=zeros(num_class,Num+1); %The number of instances belong to the ith class which have k nearest neighbors in Ci is stored in temp_Ci(i,k+1)
        temp_NCi=zeros(num_class,Num+1); %The number of instances not belong to the ith class which have k nearest neighbors in Ci is stored in temp_NCi(i,k+1)
        for i=1:num_training
            temp=zeros(1,num_class); %The number of the Num nearest neighbors of the ith instance which belong to the jth instance is stored in temp(1,j)
            neighbor_labels=[];
            for j=1:Num
                neighbor_labels=[neighbor_labels,train_target(:,Neighbors{i,1}(j))];
            end
            for j=1:num_class
                temp(1,j)=sum(neighbor_labels(j,:)==ones(1,Num));
            end
            for j=1:num_class
                if(train_target(j,i)==1)
                    temp_Ci(j,temp(j)+1)=temp_Ci(j,temp(j)+1)+1;
                else
                    temp_NCi(j,temp(j)+1)=temp_NCi(j,temp(j)+1)+1;
                end
            end
        end
        for i=1:num_class
            temp1=sum(temp_Ci(i,:));
            temp2=sum(temp_NCi(i,:));
            for j=1:Num+1
                Cond(i,j)=(Smooth+temp_Ci(i,j))/(Smooth*(Num+1)+temp1);
                CondN(i,j)=(Smooth+temp_NCi(i,j))/(Smooth*(Num+1)+temp2);
            end
        end
end