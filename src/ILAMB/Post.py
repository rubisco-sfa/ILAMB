import pylab as plt
import numpy as np
from constants import region_names

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


HEAD1 = r"""<html>
  <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="http://code.jquery.com/mobile/1.4.5/jquery.mobile-1.4.5.min.css">
    <script src="http://code.jquery.com/jquery-1.11.2.min.js"></script>
    <script src="http://code.jquery.com/mobile/1.4.5/jquery.mobile-1.4.5.min.js"></script>
    <script type="text/javascript" src="https://www.google.com/jsapi"></script>
    <script type="text/javascript">
      google.load("visualization", "1", {packages:["table"]});
      google.setOnLoadCallback(drawTable);
      function drawTable() {
        var data = new google.visualization.DataTable();
"""

HEAD2 = r"""
        var table = new google.visualization.Table(document.getElementById('table_div'));
        table.draw(data, {showRowNumber: false,allowHtml: true});
      }
    </script>"""

def ConfrontationTableGoogle(c,metrics):
    """Write out confrontation data in a HTML format"""
    def _column_sort(name,priority=["Bias","RMSE","Phase","Seasonal","Interannual","Score","Overall"]):
        """a local function to sort columns"""
        val = 1.
        for i,pname in enumerate(priority):
            if pname in name: val += 2**i
        return val

    # which metrics will we have
    models  = metrics.keys()
    models  = sorted(models,key=lambda key: key.upper())
    regions = metrics[models[0]].keys()
    regions = sorted(regions)
    header  = metrics[models[0]][regions[0]].keys()
    header  = sorted(header,key=_column_sort)

    # write out header of the html
    s  = HEAD1
    s += "        data.addColumn('string','Model');\n"
    s += "        data.addColumn('string','Data');\n"
    for region in regions:
        for h in header:
            metric = metrics[models[0]][region][h]
            unit   = metric.unit.replace(" ",r"&thinsp;").replace("-1",r"<sup>-1</sup>")
            s += """        data.addColumn('number','<span title="%s">%s [%s]</span>');\n""" % (metric.name,h,unit)
    s += "        data.addRows([\n"
    for model in models:
        s += "          ['%s'" % model
        s += """,'<a href = "%s_%s.nc" download>  [-]</a>'""" % (c.name,model)
        for region in regions:
            for h in header:
                s += ",%.03f" % metrics[model][region][h].data
        s+= "],\n"
    s += """        ]);
        var view  = new google.visualization.DataView(data);
        var rid   = document.getElementById("region").selectedIndex
"""
    lenh = len(header)
    s += "        view.setColumns([0,1"
    for i in range(lenh):
        s += ",%d*rid+%d" % (lenh,i+2)
    s += "]);"
    s += """
        var table = new google.visualization.Table(document.getElementById('table_div'));
        table.draw(view, {showRowNumber: false,allowHtml: true});
    """
    s += """    function updateImages() {
            var row = table.getSelection()[0].row;
            var rid = document.getElementById("region").selectedIndex
            var reg = document.getElementById("region").options[rid].value
            var mod = data.getValue(row, 0)
            $("#header h1 #htxt").text("%s / " + mod + " / " + reg);\n""" % c.name

    # what images do we have?
    imgs = []
    for d in c.layout:
        for img in d["plots"].keys():
            imgs.append(img)
    for img in imgs:
        if img == "compcycle":
            s += """            document.getElementById("%s").src = reg + "_%s.png"\n""" % (img,img)
        else:
            s += """            document.getElementById("%s").src = mod + "_" + reg + "_%s.png"\n""" % (img,img)


    s += """        }
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
  <body>
    <div data-role="page" id="pageone">
      <div id="header" data-role="header" data-position="fixed" data-tap-toggle="false">
	<h1><span id="htxt">%s</span></h1>
      </div>
      <select id="region" onchange="drawTable()">\n""" % c.name
    for region in regions:
        slc = ""
        if region == "global": slc = 'selected = "selected"'
        s += """        <option value="%s" %s>%s (%s)</option>\n""" % (region,slc,region,region_names[region])
    s += """      </select>
    <div id="table_div" align="center"></div>\n"""

    for sec in c.layout:
        s += """      <div data-role="collapsible" data-collapsed="false">
	<h1>%s</h1>\n""" % sec["name"]
        for plot in sec["plots"].keys():
            s += """	<table data-role="table" class="ui-responsive" id="myTable">
	  <thead>
            <tr>
	      <th align="right" width="20"><h1 id="myH1">%s</h1></th>
	      <th align="left"><img src="Benchmark_global_%s.png" id="%s" width=680 height=280 alt="Data not available"></img></th>
            </tr>
	  </thead>\n""" % (sec["plots"][plot][0],plot,plot)
            if sec["plots"][plot][1]:
                s += """         <tbody>
            <tr>
	      <th width="20"></th>
	      <th><img src="legend_%s.png" id="leg" width=680 alt="Data not available"></img></th>
            </tr>
          </tbody>\n""" % (plot)
            s += "</table>\n""" 
        s += "</div>"
    s += """
  </body>
</html>
""" 
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

def ColorBar(ax,**keywords):
    from matplotlib import colorbar,colors
    vmin  = keywords.get("vmin",None)
    vmax  = keywords.get("vmax",None)
    cmap  = keywords.get("cmap","jet")
    ticks = keywords.get("ticks",None)
    ticklabels = keywords.get("ticklabels",None)
    label = keywords.get("label",None)
    cb = colorbar.ColorbarBase(ax,cmap=cmap,
                               norm=colors.Normalize(vmin=vmin,vmax=vmax),
                               orientation='horizontal')
    cb.set_label(label)
    if ticks is not None: cb.set_ticks(ticks)
    if ticklabels is not None: cb.set_ticklabels(ticklabels)

def CompositeAnnualCycleGoogleChart(data):
    models  = data.keys()
    regions = data[models[0]].keys() 
    s = """<html>
  <head>
    <script type="text/javascript"
          src="https://www.google.com/jsapi?autoload={
            'modules':[{
              'name':'visualization',
              'version':'1',
              'packages':['corechart']
            }]
          }">
    </script>
    <script type="text/javascript">
      google.setOnLoadCallback(drawChart);
      function drawChart() {
        var data = new google.visualization.DataTable();
        data.addColumn('string', 'Month');\n"""
    for region in regions:
        for model in models:
            s += "        data.addColumn('number', '%s');\n" % model
    s += "        data.addRows([\n"
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    for i in range(12):
        s += "          ['%s'" % (months[i])
        for region in regions:
            for model in models:
                s += ",%.2f" % data[model][region].data[i]
        s += "],\n"
    s += """        ]);
        var view = new google.visualization.DataView(data);
        var region_id = 0 \n""" # document.getElementById("region").selectedIndex\n"""
    s += "        view.setColumns([0"
    lenM = len(models)
    lenR = len(regions)
    for i in range(lenM):
        s += ",region_id*%d+%d+1" % (lenR,i)
    s += """]);
        var options = {
          title: 'Annual Cycle',
          curveType: 'none',
          legend: { position: 'right' }
        };
        var chart = new google.visualization.LineChart(document.getElementById('curve_chart'));
        chart.draw(view, options);
      }
    </script>
  </head>
  <body>
    <div id="curve_chart" style="width: 900px; height: 500px"></div>
  </body>
</html>"""
    return s



def TaylorDiagram(stddev,corrcoef,refstd,fig,colors,normalize=True):
    """
    """
    from matplotlib.projections import PolarAxes
    import mpl_toolkits.axisartist.floating_axes as FA
    import mpl_toolkits.axisartist.grid_finder as GF

    # define transform
    tr = PolarAxes.PolarTransform()
    
    # correlation labels
    rlocs = np.concatenate((np.arange(10)/10.,[0.95,0.99]))
    tlocs = np.arccos(rlocs)
    gl1   = GF.FixedLocator(tlocs)
    tf1   = GF.DictFormatter(dict(zip(tlocs,map(str,rlocs))))

    # standard deviation axis extent
    if normalize:
        stddev = stddev/refstd
        refstd = 1.
    smin = 0
    smax = max(2.0,1.1*stddev.max())

    # add the curvilinear grid
    ghelper = FA.GridHelperCurveLinear(tr,
                                       extremes=(0,np.pi/2,smin,smax),
                                       grid_locator1=gl1,
                                       tick_formatter1=tf1)
    ax = FA.FloatingSubplot(fig, 111, grid_helper=ghelper)
    fig.add_subplot(ax)

    # adjust axes
    ax.axis["top"].set_axis_direction("bottom")
    ax.axis["top"].toggle(ticklabels=True,label=True)
    ax.axis["top"].major_ticklabels.set_axis_direction("top")
    ax.axis["top"].label.set_axis_direction("top")
    ax.axis["top"].label.set_text("Correlation")    
    ax.axis["left"].set_axis_direction("bottom")
    if normalize:
        ax.axis["left"].label.set_text("Normalized standard deviation")
    else:
        ax.axis["left"].label.set_text("Standard deviation")
    ax.axis["right"].set_axis_direction("top")
    ax.axis["right"].toggle(ticklabels=True)
    ax.axis["right"].major_ticklabels.set_axis_direction("left")
    ax.axis["bottom"].set_visible(False)
    ax.grid(True)
    ax = ax.get_aux_axes(tr)

    # Add reference point and stddev contour
    l, = ax.plot([0],refstd,'k*',ms=12,mew=0)
    t = np.linspace(0, np.pi/2)
    r = np.zeros_like(t) + refstd
    ax.plot(t,r, 'k--')

    # centralized rms contours
    rs,ts = np.meshgrid(np.linspace(smin,smax),
                        np.linspace(0,np.pi/2))
    rms = np.sqrt(refstd**2 + rs**2 - 2*refstd*rs*np.cos(ts))
    contours = ax.contour(ts,rs,rms,5,colors='k',alpha=0.4)
    ax.clabel(contours,fmt='%1.1f')

    # Plot data
    for i in range(len(corrcoef)):
        ax.plot(np.arccos(corrcoef[i]),stddev[i],'o',color=colors[i],mew=0)

    return l
