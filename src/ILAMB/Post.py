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
        ax.plot(np.arccos(corrcoef[i]),stddev[i],'o',color=colors[i],mew=0,ms=10)

    return ax

class HtmlFigure():

    def __init__(self,name,pattern,side=None,legend=False):

        self.name    = name
        self.pattern = pattern
        self.side    = side
        self.legend  = legend

    def generateClickRow(self):
        name = self.pattern
        for token in ['CNAME','MNAME','RNAME']:
            name = name.split(token)
            name = ("' + %s + '" % token).join(name)
        name = "'%s'" % name
        name = name.replace("'' + ","")
        code = """
          document.getElementById('%s').src =  %s""" % (self.name,name)
        return code
        
    def __str__(self):

        code = """
        <div class="outer" id="%s_div">""" % (self.name)
        if self.side is not None:
            code += """
	      <div class="inner rotate">%s</div>""" % (self.side.replace(" ","&nbsp;"))
        code += """
	      <div class="second"><img src="" id="%s" width=680 alt="Data not available"></img></div>""" % (self.name)
        if self.legend:
            if self.side is not None:
                code += """
	      <div class="inner rotate"> </div>"""
            code += """
	      <div class="second"><img src="legend_%s.png" id="leg" width=680 alt="Data not available"></img></div>""" % (self.name)
        code += """
        </div><br>"""
        return code

class HtmlLayout():
    
    def __init__(self,c,regions=None):
        
        self.c        = c
        self.metrics  = None
        self.regions  = regions
        if self.regions is not None: self.regions.sort()
        self.header   = "CNAME"
        self.figures  = {}
        self.sections = None

    def setSections(self,sections):
        
        assert type(sections) == type([])
        self.sections = sections
        for section in sections: self.figures[section] = []
        
    def addFigure(self,section,name,pattern,side=None,legend=False):
        
        assert section in self.sections
        for fig in self.figures[section]:
            if fig.name == name: return
        self.figures[section].append(HtmlFigure(name,pattern,side=side,legend=legend))
        
    def setHeader(self,header):
        
        self.header = header

    def setMetrics(self,metrics):

        self.metrics = metrics
        
    def generateMetricTable(self):

        # Local function to find how deep the metric dictionary goes
        def _findDictDepth(metrics):
            tmp   = metrics
            depth = 0
            while True:
                if type(tmp) is type({}):
                    tmp    = tmp[tmp.keys()[0]]
                    depth += 1
                else:
                    return depth

        # Sorting function
        def _sortMetrics(name,priority=["Bias","RMSE","Phase","Seasonal","Interannual","Spatial","Score","Overall"]):
            val = 1.
            for i,pname in enumerate(priority):
                if pname in name: val += 2**i
            return val

        # Convenience redefinition
        c       = self.c
        metrics = self.metrics
        
        # Grab the data
        models  = metrics.keys()
        regions = self.regions
        if regions is None: regions = ['']
        data = []
        for model in models:            
            if _findDictDepth(metrics) == 2:
                for key in metrics[model].keys():
                    if data.count(key) == 0: data.append(key)
            else:
                for region in regions:
                    for key in metrics[model][region].keys():
                        if data.count(key) == 0: data.append(key)

        # Sorts
        models.sort(key=lambda key: key.upper())
        regions.sort()
        data.sort(key=_sortMetrics)
        
        # Generate the Google DataTable Javascript code
        code = """
    <script type="text/javascript" src="https://www.google.com/jsapi"></script>
    <script type="text/javascript">
      google.load("visualization", "1", {packages:["table"]});
      google.setOnLoadCallback(drawTable);
      function drawTable() {
        var data = new google.visualization.DataTable();
        data.addColumn('string','Model');
        data.addColumn('string','Data');"""
        for region in regions:
            for header in data:
                metric = None
                if region == '':
                    if header in metrics[models[0]]:
                        metric = metrics[models[0]][header]
                else:
                    if header in metrics[models[0]][region]:              
                        metric = metrics[models[0]][region][header]
                if metric is None:
                    metric_name = ""
                    metric_unit = ""
                else:
                    metric_name = metric.name
                    metric_unit = metric.unit.replace(" ",r"&thinsp;").replace("-1",r"<sup>-1</sup>")
                code += """
        data.addColumn('number','<span title="%s">%s [%s]</span>');""" % (metric_name,header,metric_unit)
        code += """
        data.addRows(["""
        for model in models:
            code += """
          ['%s','<a href="%s_%s.nc" download>[-]</a>'""" % (model,c.name,model)
            for region in regions:
                for header in data:
                    value = ", "
                    if region == '':
                        if header in metrics[models[0]]:
                            value = ",%.03f" % metrics[model][header].data
                    else:
                        if header in metrics[models[0]][region]:  
                            value = ",%.03f" % metrics[model][region][header].data
                    code += value
            code += "],"
        code += """
        ]);"""

        # Setup the view
        code += """
        var view  = new google.visualization.DataView(data);"""
        line = str(range(2,len(data)+2))[1:]
        if regions[0] == '':
            code += """
        view.setColumns([0, 1, %s);""" % line
        else:
            line  = line.replace(", ",", %d*rid+" % len(data))
            line  = "%d*rid+2" % len(data) + line[1:]
            code += """
        var rid = document.getElementById("region").selectedIndex
        view.setColumns([0, 1, %s);""" % line

        # Draw the table
        code += """
        var table = new google.visualization.Table(document.getElementById('table_div'));
        table.draw(view, {showRowNumber: false,allowHtml: true});"""

        # clickRow feedback function
        code += """
        function clickRow() {
          var header = "%s";
          var CNAME  = "%s";
          header     = header.replace("CNAME",CNAME);"""  % (self.header,self.c.name)
        if regions[0] is not '':
            code += """
          var rid    = document.getElementById("region").selectedIndex;
          var RNAME  = document.getElementById("region").options[rid].value;
          header     = header.replace("RNAME",RNAME);"""
        code += """
          var row    = 0
          var select = table.getSelection()
          if(select.length > 0){
            row = select[0].row;
          }
          var MNAME  = data.getValue(row,0);
          header     = header.replace("MNAME",MNAME);"""
        code += """
          $("#header h1 #header_txt").text(header);"""

        if self.sections is not None:
            for section in self.sections:
                for figure in self.figures[section]:
                    code += figure.generateClickRow()
        
        code += """
        }
        google.visualization.events.addListener(table, 'select', clickRow);
        table.setSelection([{'row': 0}]);
        clickRow();
      }
    </script>"""
    
        return code

    def __str__(self):

        def _sortFigures(figure,priority=["timeint","bias","phase","shift"]):
            val = 1.
            for i,pname in enumerate(priority):
                if pname in figure.name: val += 2**i
            return val
        
        # Open the html and head
        code = """<html>
  <head>"""

        # Add needed Javascript sources
        code += """
    <link rel="stylesheet" href="http://code.jquery.com/mobile/1.4.5/jquery.mobile-1.4.5.min.css">
    <script src="http://code.jquery.com/jquery-1.11.2.min.js"></script>
    <script src="http://code.jquery.com/mobile/1.4.5/jquery.mobile-1.4.5.min.js"></script>"""

        # Add Google table of metrics
        code += self.generateMetricTable()
        
        # Add a CSS style I will use for vertical labels
        code += """
    <style type="text/css">
      .outer {
             width: 40px;
          position: relative;
           display: inline-block;
            margin: 0 15px;
      }
      .inner {
         font-size: 20px;
       font-weight: bold;
          position: absolute;
               top: 50%;
              left: 50%;
      }
            .second {
         font-size: 20px;
       font-weight: bold;
          position: relative;
              left: 40px;
      }
      .rotate {
           -moz-transform: translateX(-50%) translateY(-50%) rotate(-90deg);
        -webkit-transform: translateX(-50%) translateY(-50%) rotate(-90deg);
                transform: translateX(-50%) translateY(-50%) rotate(-90deg);
      }
    </style>"""

        # Head finished, open body and a first page
        code += """
  </head>
  <body>
    <div data-role="page" id="pageone">"""

        # Page header
        code += """
      <div id="header" data-role="header" data-position="fixed" data-tap-toggle="false">
	<h1><span id="header_txt"></span></h1>
      </div>"""

        # Add optional regions pulldown
        if self.regions is not None:
            code += """
      <select id="region" onchange="drawTable()">"""
            for r in self.regions:
                if "global" in r:
                    opt = 'selected="selected"'
                else:
                    opt = ''
                code += """
        <option value="%s" %s>%s</option>""" % (r,opt,r)
            code += """
      </select>"""

        # Add the table div
        code += """
      <div id="table_div" align="center"></div>"""

        # Build user-defined sections
        if self.sections is not None:
            for section in self.sections:
                if len(self.figures[section]) == 0: continue
                self.figures[section].sort(key=_sortFigures)
                code += """
      <div data-role="collapsible" data-collapsed="false"><h1>%s</h1>""" % section
                for figure in self.figures[section]:
                    code += "%s" % (figure)
                code += """
      </div>"""
                
        # End the first and html page
        code += """
    </div>
  </body>
</html>""" 
        return code
