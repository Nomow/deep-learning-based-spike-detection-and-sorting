function [result] = detect1(dataset, femicro)
% Band-Pass filtering in the range 300-6000Hz
Fb=300;
Fh=6000;
[b,a]=butter(1,[2*Fb/femicro 2*Fh/femicro]); % femicro: sampling frequency (24kHz)

% filtrage du signal:
LFPh=filtfilt(b,a,dataset')';

%% spike count all: detection � la Quiroga + art detection
ns=femicro/1000*3; %dur�e d'un spike en �chantillon (3ms)
T=size(LFPh,2);


Fa=4.5;
sigmas=mad(LFPh',1)/0.6745;
thQ=Fa*sigmas; % treshold

troplong=2*femicro/1000; % ??
tropgrand=20*sigmas; % max length


deltas=zeros(size(LFPh));
Ns=zeros(size(LFPh,1),1);
for imicro=1:size(LFPh,1)
    % finds indicies > treshold
    ideltas=find(abs(LFPh(imicro,:))>thQ(imicro));
    
    
    
    % removes adjacent indices
    [mini,imini]=min(diff(ideltas));
    while (mini<ns)
        [~,isuppr]=min(abs(LFPh(imicro,ideltas(imini:imini+1))));
        ideltas(imini+isuppr-1)=[];
        [mini,imini]=min(diff(ideltas));
    end
    
    % removes at the end of 
    ideltas(ideltas<=ns/2)=[];
    ideltas(ideltas>=T-ns/2)=[];
    
    % �liminer les rebonds (second grande valeur voisin (>max/2) d'un
    % max - le spike doit ressortir suffisamment dans la fen�tre
    for i=ideltas'
        ifen=LFPh(imicro,i-ns/2+1:i+ns/2);
        [maxi,imax]=max(abs(ifen));
        if (imax==15)
            if max(-sign(ifen(imax))*ifen)>abs(5*maxi/10) % si il y a rebond
                ideltas(ideltas>(i-ns/2)&ideltas<(i+ns/2+1))=[];
            end
        end
    end
    
    % �liminer les spikes positifemicro ou n�gatifemicro si minoritaires
    if (sum(LFPh(imicro,ideltas)<0)<(0*length(ideltas)/100))
        ideltas(LFPh(imicro,ideltas)<0)=[];
    end
    if (sum(LFPh(imicro,ideltas)>0)<(0*length(ideltas)/100))
        ideltas(LFPh(imicro,ideltas)>0)=[];
    end
    
    
    
    


    
    
    
    %%
    %Elimination des spikes trop grand
    tmp2=abs(LFPh(imicro,ideltas));
    itmp2=tmp2<tropgrand(imicro);
    ideltas=ideltas(itmp2);
    deltas(imicro,ideltas)=1;  
    Ns(imicro)=length(ideltas);
    
%     figure, plot(LFPh(imicro,:)); hold, 
%     plot(ideltas,LFPh(imicro,ideltas),'r*');
    
end
Ns
result = find(deltas(1,:));


end

