import pylab as plt
import numpy as np

def UseLatexPltOptions(fsize=18):
    params = {'axes.titlesize':fsize,
              'axes.labelsize':fsize,
              'font.size':fsize,
              'legend.fontsize':fsize,
              'xtick.labelsize':fsize,
              'ytick.labelsize':fsize}
    plt.rcParams.update(params)
    #plt.rc('text', usetex=True)
    #plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

def ConfrontationTableASCII(c,M):
    
    # determine header info
    head = None
    for m in M:
        if c.name in m.confrontations.keys():
            head = m.confrontations[c.name]["metric"].keys()
            break
    if head is None: return ""

    # we need to sort the header, I will use a score based on words I
    # find the in header text
    def _columnval(name):
        val = 1
        if "Score"       in name: val *= 2**4
        if "Interannual" in name: val *= 2**3
        if "RMSE"        in name: val *= 2**2
        if "Bias"        in name: val *= 2**1
        return val
    head   = sorted(head,key=_columnval)
    metric = m.confrontations[c.name]["metric"]

    # what is the longest model name?
    lenM = 0
    for m in M: lenM = max(lenM,len(m.name))
    lenM += 1

    # how long is a line?
    lineL = lenM
    for h in head: lineL += len(h)+2

    s  = "".join(["-"]*lineL) + "\n"
    s += ("{0:^%d}" % lineL).format(c.name) + "\n"
    s += "".join(["-"]*lineL) + "\n"
    s += ("{0:>%d}" % lenM).format("ModelName")
    for h in head: s += ("{0:>%d}" % (len(h)+2)).format(h)
    s += "\n" + ("{0:>%d}" % lenM).format("")
    for h in head: s += ("{0:>%d}" % (len(h)+2)).format(metric[h]["unit"])
    s += "\n" + "".join(["-"]*lineL)

    # print the table
    s += ("\n{0:>%d}" % lenM).format("Benchmark")
    for h in head:
        if h in c.metric.keys():
            s += ("{0:>%d,.3f}" % (len(h)+2)).format(c.metric[h]["var"])
        else:
            s += ("{0:>%d}" % (len(h)+2)).format("~")

    for m in M:
        s += ("\n{0:>%d}" % lenM).format(m.name)
        if c.name in m.confrontations.keys():
            for h in head: s += ("{0:>%d,.3f}" % (len(h)+2)).format(m.confrontations[c.name]["metric"][h]["var"])
        else:
            for h in head: s += ("{0:>%d}" % (len(h)+2)).format("~")
    return s

def ConfrontationTableGoogle(c,M,regions=["global"]):
    from constants import region_names
    # determine header info
    head    = None
    for m in M:
        if c.name in m.confrontations.keys():
            head    = m.confrontations[c.name]["metric"].keys()
            break
    if head is None: return ""

    # we need to sort the header, I will use a score based on words I
    # find the in header text
    def _columnval(name):
        val = 1
        if "Score"       in name: val *= 2**4
        if "Interannual" in name: val *= 2**3
        if "RMSE"        in name: val *= 2**2
        if "Bias"        in name: val *= 2**1
        return val
    head   = sorted(head,key=_columnval)
    metric = m.confrontations[c.name]["metric"]

    s  = """
<html>
  <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="http://code.jquery.com/mobile/1.4.5/jquery.mobile-1.4.5.min.css">
    <script src="http://code.jquery.com/jquery-1.11.2.min.js"></script>
    <script src="http://code.jquery.com/mobile/1.4.5/jquery.mobile-1.4.5.min.js"></script>
    <script type="text/javascript" src="https://www.google.com/jsapi"></script>
    <script type="text/javascript">
      google.load("visualization", "1", {packages:["table"]});
      google.setOnLoadCallback(draw%sTable);
""" % c.name
    s += "      function draw%sTable() {\n" % c.name
    s += "        var data = new google.visualization.DataTable();\n"
    s += "        data.addColumn('string','Model');\n"
    for h in head: 
        unit = metric[h]["unit"].replace(" ",r"&thinsp;").replace("-1",r"<sup>-1</sup>")
        s += """        data.addColumn('number','<span title="%s">%s [%s]</span>');\n""" % (metric[h]["desc"],h,unit)
    s += "        data.addRows(%d);\n" % (len(M)+1)   
 
    row = 0
    s   += "        data.setCell(%d,0,'Benchmark');" % (row)
    col = 0
    for h in head:
        col += 1
        if h in c.metric.keys():
            s += "data.setCell(%d,%d,%.03f); " % (row,col,c.metric[h]["var"])
        else:
            s += "data.setCell(%d,%d,null); " % (row,col)
    s += "\n"
    for m in M:
        row += 1
        col  = 0
        s   += "        data.setCell(%d,0,'%s'); " % (row,m.name)
        if c.name in m.confrontations.keys():
            for h in head: 
                col += 1
                s   += "data.setCell(%d,%d,%.03f); " % (row,col,m.confrontations[c.name]["metric"][h]["var"])
        else:
            for h in head:
                col += 1
                s   += "data.setCell(%d,%d,null); " % (row,col)
        s += "\n"
    s += """  
        var table = new google.visualization.Table(document.getElementById('table_div'));
        table.draw(data, {showRowNumber: false,allowHtml: true});

        function updateImages() {
          try {
            var row = table.getSelection()[0].row;
          }
          catch(err) {
            var row = 0;
          }
          var reg = document.getElementById("region").options[document.getElementById("region").selectedIndex].value
          var mod = data.getValue(row, 0)
          $("#header h1 #htxt").text("GPPFluxnetGlobalMTE / " + mod + " / " + reg);
          document.getElementById("img"  ).src = mod + "_"      + reg + ".png"
          document.getElementById("bias" ).src = mod + "_Bias_" + reg + ".png"
          document.getElementById("mean" ).src = mod + "_Mean.png"
          document.getElementById("cycle").src = mod + "_Cycle.png"
          document.getElementById("peak" ).src = mod + "_Peak_" + reg + ".png"
          document.getElementById("bpeak").src = "Benchmark_Peak_" + reg + ".png"
          document.getElementById("pstd" ).src = mod + "_Pstd_"  + reg + ".png"
          document.getElementById("shift").src = mod + "_Shift_"+ reg + ".png"
        }

        google.visualization.events.addListener(table, 'select', updateImages);

        table.setSelection([{'row': 0}]);
        updateImages();
      }
    </script>
    <style>
      #myH1 {
        transform: 
          translate(0px, 140px)
          rotate(270deg);
        width: 20px;
      }
    </style>
  </head>
  <body>
    <div data-role="page" id="pageone">
      <div id="header" data-role="header" data-position="fixed" data-tap-toggle="false">
	<h1><span id="htxt">GPPFluxnetGlobalMTE</span></h1>
      </div>

      <select id="region" onchange="drawGPPFluxnetGlobalMTETable()">
"""
    for region in regions:
        s += '        <option value="%s">%s (%s)</option>\n' % (region,region_names[region],region)
    s += """
      </select>
      <div id="table_div" align="center"></div>

      <div data-role="collapsible" data-collapsed="false">
	<h1>Temporally integrated period mean</h1>
	<table data-role="table" class="ui-responsive" id="myTable">
	  <thead>
            <tr>
	      <th align="right" width="20"><h1 id="myH1">MEAN</h1></th>
	      <th align="left"><img src="Benchmark_global.png" id="img" width=680 height=280 alt="Data not available"></img></th>
	      <th align="right" width="20"><h1 id="myH1">BIAS</h1></th>
	      <th align="left"><img src="" id="bias" width=680 height=280 alt="Data not available"></img><br></th>
            </tr>
	  </thead>
         <tbody>
            <tr>
	      <th width="20"></th>
	      <th><img src="mean_legend.png" id="leg" width=680 height=102 alt="Data not available"></img></th>
	      <th width="20"></th>
	      <th><img src="bias_legend.png" id="leg" width=680 height=102 alt="Data not available"></img></th>
            </tr>
          </tbody>
	</table>
      </div>
      <div data-role="collapsible" data-collapsed="false">
	<h1>Spatially integrated regional mean</h1>
	<table data-role="table" class="ui-responsive" id="myTable">
	  <thead>
            <tr>
	      <th align="right" width="20"><h1 id="myH1">MEAN</h1></th>
	      <th align="left"><img src="" id="mean" width=680 height=280 alt="Data not available"></img></th>
            </tr>
	  </thead>
	</table>
      </div>
      <div data-role="collapsible" data-collapsed="false">
	<h1>Annual cycle</h1>
	<table data-role="table" class="ui-responsive" id="myTable">
	  <thead>
            <tr>
	      <th align="right" width="20"><h1 id="myH1">MEAN</h1></th>
	      <th align="left"><img src="" id="cycle" width=680 height=280 alt="Data not available"></img></th>
            </tr>
	  </thead>
	</table>
      </div>
      <div data-role="collapsible" data-collapsed="false">
	<h1>Phase</h1>
	<table data-role="table" class="ui-responsive" id="myTable">
         <thead>
            <tr>
	      <th align="right" width="20"><h1 id="myH1">PEAK</h1></th>
	      <th align="left"><img src="Benchmark_Peak_global.png" id="peak" width=680 height=280 alt="Data not available"></img></th>
	      <th align="right" width="20"><h1 id="myH1">OBS</h1></th>
	      <th align="left"><img src="Benchmark_Peak_global.png" id="bpeak" width=680 height=280 alt="Data not available"></img></th>
            </tr>
         </thead>
         <tbody>
            <tr>
	      <th width="20"></th>
	      <th><img src="peak_legend.png" id="leg" width=680 height=102 alt="Data not available"></img></th>
	      <th width="20"></th>
	      <th><img src="peak_legend.png" id="leg" width=680 height=102 alt="Data not available"></img></th>
            </tr>
            <tr>
	      <th align="right" width="20"><h1 id="myH1">STDEV</h1></th>
	      <th align="left"><img src="" id="pstd" width=680 height=280 alt="Data not available"></img></th>
	      <th align="right" width="20"><h1 id="myH1">SHIFT</h1></th>
	      <th align="left"><img src="" id="shift" width=680 height=280 alt="Data not available"></img></th>
            </tr>
            <tr>
	      <th width="20"></th>
	      <th><img src="pstd_legend.png" id="leg" width=680 height=102 alt="Data not available"></img></th>
	      <th width="20"></th>
	      <th><img src="shift_legend.png" id="leg" width=680 height=102 alt="Data not available"></img></th>
            </tr>
          </tbody>
	</table>
      </div>

  </body>
</html>"""
    return s

def GlobalPlot(lat,lon,var,ax,region="global.large",shift=False,**keywords):
    """

    """
    from mpl_toolkits.basemap import Basemap
    from constants import regions

    vmin  = keywords.get("vmin",None)
    vmax  = keywords.get("vmax",None)
    cmap  = keywords.get("cmap","jet")
    ticks = keywords.get("ticks",None)
    ticklabels = keywords.get("ticklabels",None)
    unit  = keywords.get("unit",None)

    # aspect ratio stuff
    lats,lons = regions[region]
    lats = np.asarray(lats); lons = np.asarray(lons)
    dlat,dlon = lats[1]-lats[0],lons[1]-lons[0]
    fsize = ax.get_figure().get_size_inches()
    figure_ar = fsize[1]/fsize[0]
    scale = figure_ar*dlon/dlat
    if scale >= 1.:
        lats[1] += 0.5*dlat*(scale-1.)
        lats[0] -= 0.5*dlat*(scale-1.)
    else:
        scale = 1./scale
        lons[1] += 0.5*dlon*(scale-1.)
        lons[0] -= 0.5*dlon*(scale-1.)
    lats = lats.clip(-90,90)
    lons = lons.clip(-180,180)

    bmap = Basemap(projection='cyl',
                   llcrnrlon=lons[ 0],llcrnrlat=lats[ 0],
                   urcrnrlon=lons[-1],urcrnrlat=lats[-1],
                   resolution='c',ax=ax)
    if shift:
        nroll = np.argmin(np.abs(lon-180))
        alon  = np.roll(lon,nroll); alon[:nroll] -= 360
        tmp   = np.roll(var,nroll,axis=1)
    else:
        alon = lon
        tmp  = var
    x,y = bmap(alon,lat)
    ax  = bmap.pcolormesh(x,y,tmp,zorder=2,vmin=vmin,vmax=vmax,cmap=cmap)

    bmap.drawcoastlines(linewidth=0.2,color="darkslategrey")

def ColorBar(var,ax,**keywords):
    from matplotlib import colorbar,colors
    vmin  = keywords.get("vmin",None)
    vmax  = keywords.get("vmax",None)
    cmap  = keywords.get("cmap","jet")
    ticks = keywords.get("ticks",None)
    ticklabels = keywords.get("ticklabels",None)
    label = keywords.get("label",None)
    if vmin is None: vmin = np.ma.min(var)
    if vmax is None: vmax = np.ma.max(var)
    cb = colorbar.ColorbarBase(ax,cmap=cmap,
                               norm=colors.Normalize(vmin=vmin,vmax=vmax),
                               orientation='horizontal')
    cb.set_label(label)
    if ticks is not None: cb.set_ticks(ticks)
    if ticklabels is not None: cb.set_ticklabels(ticklabels)
