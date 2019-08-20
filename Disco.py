def distance_corr(yqcd,mqcd,normedweight,power=1):
    xx = yqcd.view(-1, 1).repeat(1, len(yqcd)).view(len(yqcd),len(yqcd))
    yy = yqcd.repeat(len(yqcd),1).view(len(yqcd),len(yqcd))
    amat=(xx-yy).abs()

    xx = mqcd.view(-1, 1).repeat(1, len(mqcd)).view(len(mqcd),len(mqcd))
    yy = mqcd.repeat(len(mqcd),1).view(len(mqcd),len(mqcd))
    bmat=(xx-yy).abs()

    amatavg=torch.mean(amat*normedweight,dim=1)
    Amat=amat-amatavg.repeat(len(yqcd),1).view(len(yqcd),len(yqcd))\
        -amatavg.view(-1, 1).repeat(1, len(yqcd)).view(len(yqcd),len(yqcd))\
        +torch.mean(amatavg*normedweight)

    bmatavg=torch.mean(bmat*normedweight,dim=1)
    Bmat=bmat-bmatavg.repeat(len(mqcd),1).view(len(mqcd),len(mqcd))\
        -bmatavg.view(-1, 1).repeat(1, len(mqcd)).view(len(mqcd),len(mqcd))\
        +torch.mean(bmatavg*normedweight)

    ABavg=torch.mean(Amat*Bmat*normedweight,dim=1)
    AAavg=torch.mean(Amat*Amat*normedweight,dim=1)
    BBavg=torch.mean(Bmat*Bmat*normedweight,dim=1)

    if(power==1):
        dCorr=(torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight)))
    elif(power==2):
        dCorr=(torch.mean(ABavg*normedweight))**2/(torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))
    else:
        dCorr=(torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight)))**power
    return dCorr


