function [result] = detect3(dataset,femicro)

Fb=300;
Fh=6000;
[b,a]=butter(1,[2*Fb/femicro 2*Fh/femicro]); % femicro: sampling frequency (24kHz)

% filtrage du signal:
LFPh=filtfilt(b,a,dataset')';
    
    Fa=5;
     sigmas=mad(LFPh',1)/0.6745;
     thQ=Fa*sigmas; % detection threshold

     troplong=2*femicro/1000; % too long threshold
     tropgrand=20*sigmas; % too big threshold
     tropproche=32; % samples between two spikes (too close threshold)


     rastMUA2=zeros(size(LFPh));
     for imicro=1:size(LFPh,1),
         irastMUA=abs(LFPh(imicro,:))>thQ(imicro);
         tmp=diff(irastMUA);
         inpic=find(tmp==1)+1;
         outpic=find(tmp==-1);
         itmp=(outpic-inpic)<troplong;
         inpic=inpic(itmp);
         outpic=outpic(itmp);
         npics=length(inpic);
         timemax=zeros(1,npics);
         clear tmp2 timemax indmax
         for ipics=1:npics,
             [tmp2(ipics),indmax(ipics)]=max(abs(LFPh(imicro,inpic(ipics):outpic(ipics))));
             timemax(ipics)=inpic(ipics)+indmax(ipics)-1;
         end
         if npics ~= 0
         itmp2=tmp2<tropgrand(imicro);
         rastMUA2(imicro,timemax(itmp2))=1;
         itmp3=(inpic(2:end)-outpic(1:end-1))<tropproche;
         imua=unique([(find(itmp3)+1),find(itmp3)]); % indices of too 
         end
     end
  result1 = find(rastMUA2 == 1)

