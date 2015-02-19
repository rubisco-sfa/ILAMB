import pylab as plt

def UseLatexPltOptions(fsize=18):
    params = {'axes.titlesize':fsize,
              'axes.labelsize':fsize,
              'font.size':fsize,
              'legend.fontsize':fsize,
              'xtick.labelsize':fsize,
              'ytick.labelsize':fsize}
    plt.rcParams.update(params)
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

def ConfrontationTableASCII(cname,M):
    
    # determine header info
    head = None
    for m in M:
        if cname in m.confrontations.keys():
            head = m.confrontations[cname]["metric"].keys()
            break
    assert head is not None

    # we need to sort the header, I will use a score based on words I
    # find the in header text
    def _columnval(name):
        val = 1
        if "Score"       in name: val *= 2**4
        if "Interannual" in name: val *= 2**3
        if "RMSE"        in name: val *= 2**2
        if "Bias"        in name: val *= 2**1
        return val
    head = sorted(head,key=_columnval)

    # what is the longest model name?
    lenM = 0
    for m in M: lenM = max(lenM,len(m.name))
    lenM += 1

    # how long is a line?
    lineL = lenM
    for h in head: lineL += len(h)+2

    s  = "".join(["-"]*lineL) + "\n"
    s += ("{0:>%d}" % lenM).format("ModelName")
    for h in head: s += ("{0:>%d}" % (len(h)+2)).format(h)
    s += "\n" + "".join(["-"]*lineL)

    # print the table
    for m in M:
        s += ("\n{0:>%d}" % lenM).format(m.name)
        if cname in m.confrontations.keys():
            for h in head: s += ("{0:>%d,.3f}" % (len(h)+2)).format(m.confrontations[cname]["metric"][h])
        else:
            for h in head: s += ("{0:>%d}" % (len(h)+2)).format("~")
    return s
