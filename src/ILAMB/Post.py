import pylab as plt
import numpy as np
from constants import region_names
import re

def UseLatexPltOptions(fsize=18):
    params = {'axes.titlesize':fsize,
              'axes.labelsize':fsize,
              'font.size':fsize,
              'legend.fontsize':fsize,
              'xtick.labelsize':fsize,
              'ytick.labelsize':fsize}
    plt.rcParams.update(params)

def UnitStringToMatplotlib(unit,add_carbon=False):
    # raise exponents using Latex
    match = re.findall("(-\d)",unit)
    for m in match: unit = unit.replace(m,"$^{%s}$" % m)
    # add carbon symbol to all mass units
    if add_carbon:
        match = re.findall("(\D*g)",unit)
        for m in match: unit = unit.replace(m,"%s C " % m)
    return unit

def GlobalPlot(lat,lon,var,ax,region="global",shift=False,**keywords):
    """Use basemap to plot data on the globe.

    Parameters
    ----------
    lat : numpy.ndarray
        a 1D array of latitudes
    lon : numpy.ndarray
        a 1D array of longitudes
    var : numpy.ndarray
        a 2D array of data
    ax : matplotlib.axes._subplots.AxesSubplot
        the matplotlib axes object onto which you wish to plot the variable
    region : str, optional
        the region on which to plot
    shift : bool, optional
        enable to move the first column of data to the international dateline
    vmin : float, optional
        the minimum plotted value
    vmax : float, optional
        the maximum plotted value
    cmap : str, optional
        the name of the colormap to be used in plotting the spatial variable
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

    if shift:
        nroll = np.argmin(np.abs(lon-180))
        alon  = np.roll(lon,nroll); alon[:nroll] -= 360
        tmp   = np.roll(var,nroll,axis=1)
    else:
        alon = lon
        tmp  = var

    #if region is None or region == "global":
    #if False:
    #    bmap = Basemap(projection = 'robin',
    #                   lon_0      = 0.,
    #                   resolution = 'c',
    #                   ax         = ax)        
    #    x,y = np.meshgrid(alon,lat)
    #    ax  = bmap.pcolormesh(x,y,tmp,latlon=True,vmin=vmin,vmax=vmax,cmap=cmap)
    if region == "arctic":
        mp = Basemap(projection='npstere',boundinglat=60,lon_0=180,resolution='c')
        #mp = Basemap(projection='ortho',lat_0=90.,lon_0=180.,resolution='c')
        X,Y = np.meshgrid(lat,alon,indexing='ij')
        mp.pcolormesh(Y,X,tmp,latlon=True,cmap=cmap,vmin=vmin,vmax=vmax)
        mp.drawlsmask(land_color='lightgrey',ocean_color='grey',lakes=True)
    else:
        bmap = Basemap(projection = 'cyl',
                       llcrnrlon  = lons[ 0],
                       llcrnrlat  = lats[ 0],
                       urcrnrlon  = lons[-1],
                       urcrnrlat  = lats[-1],
                       resolution = 'c',
                       ax         = ax)  
        ax  = bmap.pcolormesh(alon,lat,tmp,latlon=True,vmin=vmin,vmax=vmax,cmap=cmap)
        bmap.drawcoastlines(linewidth=0.2,color="darkslategrey")

def ColorBar(ax,**keywords):
    """Plot a colorbar.

    We plot colorbars separately so they can be rendered once and used
    for multiple plots.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        the matplotlib axes object onto which you wish to plot the variable
    vmin : float, optional
        the minimum plotted value
    vmax : float, optional
        the maximum plotted value
    cmap : str, optional
        the name of the colormap to be used in plotting the spatial variable
    label : str, optional
        the text which appears with the colorbar

    """
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
    """Plot a Taylor diagram.

    This is adapted from the code by Yannick Copin found here:

    https://gist.github.com/ycopin/3342888
    
    Parameters
    ----------
    stddev : numpy.ndarray
        an array of standard deviations
    corrcoeff : numpy.ndarray
        an array of correlation coefficients
    refstd : float
        the reference standard deviation
    fig : matplotlib figure
        the matplotlib figure
    colors : array
        an array of colors for each element of the input arrays
    normalize : bool, optional
        disable to skip normalization of the standard deviation

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
    # Plot data
    corrcoef = corrcoef.clip(-1,1)
    for i in range(len(corrcoef)):
        ax.plot(np.arccos(corrcoef[i]),stddev[i],'o',color=colors[i],mew=0,ms=8)
            
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


    return ax

class HtmlFigure():

    def __init__(self,name,pattern,side=None,legend=False,benchmark=False):

        self.name      = name
        self.pattern   = pattern
        self.side      = side
        self.legend    = legend
        self.benchmark = benchmark
        
    def generateClickRow(self,allModels=False):
        name = self.pattern
        if allModels: name = name.replace(self.name,"PNAME")
        for token in ['CNAME','MNAME','RNAME','PNAME']:
            name = name.split(token)
            name = ("' + %s + '" % token).join(name)
        name = "'%s'" % name
        name = name.replace("'' + ","")
        code = """
          document.getElementById('%s').src =  %s""" % (self.name,name)
        if self.benchmark:
            name = self.pattern.replace('MNAME','Benchmark')
            for token in ['CNAME','MNAME','RNAME']:
                name = name.split(token)
                name = ("' + %s + '" % token).join(name)
            name = "'%s'" % name
            name = name.replace("'' + ","")
            code += """
          document.getElementById('benchmark_%s').src =  %s""" % (self.name,name)
        return code

    def __str__(self):

        if self.benchmark:
            code = """
        <div class="benchmark" id="%s_div">""" % (self.name)
        else:
            code = """
        <div class="outer" id="%s_div">""" % (self.name)

        if self.benchmark:
            code += """
              <div class="second">
                 <img src="" id="benchmark_%s" alt="Data not available"></img>
                 <img src="" id="%s" alt="Data not available"></img>
              </div>""" % (self.name,self.name)
        else:            
            if self.side is not None:
                code += """
              <div class="inner rotate">%s</div>""" % (self.side.replace(" ","&nbsp;"))
            code += """
              <div class="second"><img src="" id="%s" alt="Data not available"></img></div>""" % (self.name)
            if self.legend:
                if self.side is not None:
                    code += """
              <div class="inner rotate"> </div>"""
                code += """
              <div class="second"><img src="legend_%s.png" id="leg"  alt="Data not available"></img></div>""" % (self.name)
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
        self.priority = ["Bias","RMSE","Phase","Seasonal","Spatial","Interannual","Score","Overall"]
        
    def setSections(self,sections):

        assert type(sections) == type([])
        self.sections = sections
        for section in sections: self.figures[section] = []

    def addFigure(self,section,name,pattern,side=None,legend=False,benchmark=False):

        assert section in self.sections
        for fig in self.figures[section]:
            if fig.name == name: return
        self.figures[section].append(HtmlFigure(name,pattern,side=side,legend=legend,benchmark=benchmark))

    def setHeader(self,header):

        self.header = header

    def setMetrics(self,metrics):

        self.metrics = metrics

    def setMetricPriority(self,priority):

        self.priority = priority
        
    def generateMetricTable(self):

        # Sorting function
        def _sortMetrics(name,priority=self.priority):
            val = 1.
            for i,pname in enumerate(priority):
                if pname in name: val += 2**i
            return val

        # Convenience redefinition
        c       = self.c
        metrics = self.metrics
        regions = self.regions
        models  = metrics.keys()
        
        # Grab the data
        if regions is None: regions = ['']
        data = []
        for model in models:
            for region in regions:
                if not metrics[model].has_key(region): continue
                for key in metrics[model][region].keys():
                    if data.count(key) == 0: data.append(key)

        # Sorts
        models.sort(key=lambda key: key.upper())
        try:
            models.insert(0,models.pop(models.index("Benchmark")))
        except:
            pass
        regions.sort()
        data.sort(key=_sortMetrics)

        # List of plots for the 'All Models' page
        plots = []
        bench = []
        if self.sections is not None:
            for section in self.sections:
                if len(self.figures[section]) == 0: continue
                for figure in self.figures[section]:
                    if figure.name in ['timeint','bias','phase','shift']:
                        if figure not in plots: plots.append(figure)
                    if "benchmark" in figure.name:
                        if figure.name not in bench: bench.append(figure.name)
        nobench = [plot.name for plot in plots if "benchmark_%s" % (plot.name) not in bench]        
        
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
                    if header in metrics[models[1]]:
                        metric = metrics[models[1]][header]
                else:
                    if header in metrics[models[1]][region]:
                        metric = metrics[models[1]][region][header]
                if metric is None:
                    metric_name = ""
                    metric_unit = ""
                else:
                    metric_name = metric.name
                    metric_unit = metric.unit.replace(" " ,r"&thinsp;")
                    metric_unit = metric.unit.replace("-1",r"<sup>-1</sup>")
                    metric_unit = metric.unit.replace("-2",r"<sup>-2</sup>")
                code += """
        data.addColumn('number','<span title="%s">%s [%s]</span>');""" % (metric_name,header,metric_unit)
        code += """
        data.addRows(["""
        for model in models:
            code += """
          ['%s','<a href="%s_%s.nc" download>[-]</a>'""" % (model,c.name,model)
            for region in regions:
                for header in data:
                    value = ", null"
                    if region == '':
                        assert False
                    else:
                        if metrics[model].has_key(region):
                            if header in metrics[model][region]:
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
          header     = header.replace("CNAME",CNAME);"""  % (self.header,self.c.longname.replace("/"," / "))
        if regions[0] is not '':
            code += """
          var rid    = document.getElementById("region").selectedIndex;
          var RNAME  = document.getElementById("region").options[rid].value;
          header     = header.replace("RNAME",RNAME);"""
        code += """
          var select = table.getSelection()
          row = select[0].row;
          if (row == 0) {
            table.setSelection([{'row': 1}]);
            clickRow();
            return;
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

        code += """
    
    <script>
      function select2() {
        var header = "%s";
        var CNAME  = "%s";
        header     = header.replace("CNAME",CNAME);
        var rid    = document.getElementById("region2").selectedIndex;
        var RNAME  = document.getElementById("region2").options[rid].value;
        var pid    = document.getElementById("plot"  ).selectedIndex;
        var PNAME  = document.getElementById("plot"  ).options[pid].value;
        header     = header.replace("RNAME",RNAME);
        $("#header h1 #header_txt").text(header);""" % (self.header.replace(" / MNAME",""),self.c.longname.replace("/"," / "))


        code += """
        if(%s){
          document.getElementById("Benchmark_div").style.display = 'none'
        }else{
          document.getElementById("Benchmark_div").style.display = 'block'
          document.getElementById('Benchmark').src = 'Benchmark_' + RNAME + '_' + PNAME + '.png'
        }
        document.getElementById('legend').src = 'legend_' + PNAME + '.png'""" % (" || ".join(['PNAME == "%s"' % n for n in nobench]))

        for model in models:
            code += """
        document.getElementById('%s').src = '%s_' + RNAME + '_' + PNAME + '.png'""" % (model,model)
        
        code += """
      }
    </script>
    <script>
      $(document).on('pageshow', '[data-role="page"]', function(){ 
        select2()
      });
    </script>"""

        return code

    def __str__(self):
        
        def _sortFigures(figure,priority=["benchmark_timeint","timeint","bias","benchmark_phase","phase","shift","spatial_variance","spaceint","cycle","compcycle"]):
            val = 1.
            for i,pname in enumerate(priority):
                if pname == figure.name: val += 2**i
            return val

        # Open the html and head
        code = """<html>
  <head>"""

        # Add needed Javascript sources
        code += """
    <link rel="stylesheet" href="http://code.jquery.com/mobile/1.4.5/jquery.mobile-1.4.5.min.css">
    <script src="http://code.jquery.com/jquery-1.11.2.min.js"></script>
    <script>
      $(document).bind('mobileinit',function(){
        $.mobile.changePage.defaults.changeHash = false;
        $.mobile.hashListeningEnabled = false;
        $.mobile.pushStateEnabled = false;
      });
    </script>
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
      .benchmark {
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
    <div data-role="page" id="page1">"""

        # Page header
        code += """
      <div id="header" data-role="header" data-position="fixed" data-tap-toggle="false">
        <h1><span id="header_txt"></span></h1>
	<div data-role="navbar">
	  <ul>
	    <li><a href="#page1" class="ui-btn-active ui-state-persist">Single Model</a></li>
	    <li><a href="#page2">All Models</a></li>
	  </ul>
	</div>
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
    </div>"""

        # Second page
        code += """
    <div data-role="page" id="page2">
      <div id="header" data-role="header" data-position="fixed" data-tap-toggle="false">
        <h1><span id="header_txt"></span></h1>
	<div data-role="navbar">
	  <ul>
	    <li><a href="#page1">Single Model</a></li>
	    <li><a href="#page2" class="ui-btn-active ui-state-persist">All Models</a></li>
	  </ul>
	</div>
      </div>"""
        
        # Add optional regions pulldown
        if self.regions is not None:
            code += """
      <select id="region2" onchange="select2()">"""
            for r in self.regions:
                if "global" in r:
                    opt = 'selected="selected"'
                else:
                    opt = ''
                code += """
        <option value="%s" %s>%s</option>""" % (r,opt,r)
            code += """
      </select>"""

        # Add a plot for each model
        models = self.metrics.keys()
        models.sort(key=lambda key: key.upper())
        try:
            models.insert(0,models.pop(models.index("Benchmark")))
        except:
            pass

        # Which plots to add?
        figs = []
        if self.sections is not None:
            for section in self.sections:
                if len(self.figures[section]) == 0: continue
                for figure in self.figures[section]:
                    if (figure.name in ['timeint','bias','phase','shift','whittaker'] or
                        "rel_" in figure.name):
                        if figure not in figs: figs.append(figure)

        code += """
      <select id="plot" onchange="select2()">"""
        from constants import space_opts
        for f in figs:
            opt = ''
            if "timeint" is f.name: opt = 'selected="selected"'
            name = ""
            if space_opts.has_key(f.name): name = space_opts[f.name]["name"]
            if "rel_" in f.name: name = f.name.replace("rel_","Relationship with ")
            if "whittaker" == f.name: name = "Whittaker"
            code += """
        <option value="%s" %s>%s</option>""" % (f.name,opt,name)
        code += """
      </select>"""

        if len(figs) > 0:
            fig = figs[0]
            rem_legend = fig.legend; fig.legend = False
            rem_side   = fig.side;   fig.side   = "MNAME"
            img = "%s" % (fig)
            img = img.replace("%s" % fig.name,"MNAME")
            fig.legend = rem_legend
            fig.side   = rem_side
            for model in models:
                code += img.replace("MNAME",model)

            code += """
        <div class="outer" id="legend_div">
              <div class="inner rotate"> </div>
              <div class="second"><img src="" id="legend" alt="Data not available"></img></div>
        </div><br>"""
        
        # close page 2 and end
        code += """
    </div>
  </body>
</html>"""
        return code

def RegisterCustomColormaps():
    """Adds the 'stoplight' and 'RdGn' colormaps to matplotlib's database

    """
    import colorsys as cs
    
    # stoplight colormap
    Rd1    = [1.,0.,0.]; Rd2 = Rd1
    Yl1    = [1.,1.,0.]; Yl2 = Yl1
    Gn1    = [0.,1.,0.]; Gn2 = Gn1
    val    = 0.65
    Rd1    = cs.rgb_to_hsv(Rd1[0],Rd1[1],Rd1[2])
    Rd1    = cs.hsv_to_rgb(Rd1[0],Rd1[1],val   )
    Yl1    = cs.rgb_to_hsv(Yl1[0],Yl1[1],Yl1[2])
    Yl1    = cs.hsv_to_rgb(Yl1[0],Yl1[1],val   )
    Gn1    = cs.rgb_to_hsv(Gn1[0],Gn1[1],Gn1[2])
    Gn1    = cs.hsv_to_rgb(Gn1[0],Gn1[1],val   )
    p      = 0
    level1 = 0.5
    level2 = 0.75
    RdYlGn = {'red':   ((0.0     , 0.0   ,Rd1[0]),
                        (level1-p, Rd2[0],Rd2[0]),
                        (level1+p, Yl1[0],Yl1[0]),
                        (level2-p, Yl2[0],Yl2[0]),
                        (level2+p, Gn1[0],Gn1[0]),
                        (1.00    , Gn2[0],  0.0)),
              
              'green': ((0.0     , 0.0   ,Rd1[1]),
                        (level1-p, Rd2[1],Rd2[1]),
                        (level1+p, Yl1[1],Yl1[1]),
                        (level2-p, Yl2[1],Yl2[1]),
                        (level2+p, Gn1[1],Gn1[1]),
                        (1.00    , Gn2[1],  0.0)),
              
              'blue':  ((0.0     , 0.0   ,Rd1[2]),
                        (level1-p, Rd2[2],Rd2[2]),
                        (level1+p, Yl1[2],Yl1[2]),
                        (level2-p, Yl2[2],Yl2[2]),
                        (level2+p, Gn1[2],Gn1[2]),
                        (1.00    , Gn2[2],  0.0))}
    plt.register_cmap(name='stoplight', data=RdYlGn)
    
    # RdGn colormap
    val = 0.8
    Rd  = cs.rgb_to_hsv(1,0,0)
    Rd  = cs.hsv_to_rgb(Rd[0],Rd[1],val)
    Gn  = cs.rgb_to_hsv(0,1,0)
    Gn  = cs.hsv_to_rgb(Gn[0],Gn[1],val)
    RdGn = {'red':   ((0.0, 0.0,   Rd[0]),
                      (0.5, 1.0  , 1.0  ),
                      (1.0, Gn[0], 0.0  )),
            'green': ((0.0, 0.0,   Rd[1]),
                      (0.5, 1.0,   1.0  ),
                      (1.0, Gn[1], 0.0  )),
            'blue':  ((0.0, 0.0,   Rd[2]),
                      (0.5, 1.0,   1.0  ),
                      (1.0, Gn[2], 0.0  ))}
    plt.register_cmap(name='RdGn', data=RdGn)


def BenchmarkSummaryFigure(models,variables,data,figname,vcolor=None):
    """Creates a summary figure for the benchmark results contained in the
    data array.

    Parameters
    ----------
    models : list
        a list of the model names 
    variables : list
        a list of the variable names
    data : numpy.ndarray or numpy.ma.ndarray
        data scores whose shape is ( len(variables), len(models) )
    figname : str
        the full path of the output file to write
    vcolor : list, optional
        an array parallel to the variables array containing background 
        colors for the labels to be displayed on the y-axis.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # data checks
    assert  type(models)    is type(list())
    assert  type(variables) is type(list())
    assert (type(data)      is type(np   .empty(1)) or
            type(data)      is type(np.ma.empty(1)))
    assert data.shape[0] == len(variables)
    assert data.shape[1] == len(models   )
    assert  type(figname)   is type("")
    if vcolor is not None:
        assert type(vcolor) is type(list())
        assert len(vcolor) == len(variables)
        
    # define some parameters
    nmodels    = len(models)
    nvariables = len(variables)
    w          = max((nmodels-3.)/(14.-3.)*(9.5-5.08)+5.08,7.) # heuristic for figure size
    h          = 8.
    bad        = 0.5
    if "stoplight" not in plt.colormaps(): RegisterCustomColormaps()
    
    # plot the variable scores
    fig,ax = plt.subplots(figsize=(w,h),ncols=2,tight_layout=True)
    cmap   = plt.get_cmap('stoplight')
    cmap.set_bad('k',bad)
    qc     = ax[0].pcolormesh(np.ma.masked_invalid(data[::-1,:]),cmap=cmap,vmin=0,vmax=1,linewidth=0)
    div    = make_axes_locatable(ax[0])
    fig.colorbar(qc,
                 ticks=(0,0.25,0.5,0.75,1.0),
                 format="%g",
                 cax=div.append_axes("bottom", size="5%", pad=0.05),
                 orientation="horizontal",
                 label="Variable Score")
    plt.tick_params(which='both', length=0)
    ax[0].xaxis.tick_top()
    ax[0].set_xticks     (np.arange(nmodels   )+0.5)
    ax[0].set_xticklabels(models,rotation=90)
    ax[0].set_yticks     (np.arange(nvariables)+0.5)
    ax[0].set_yticklabels(variables[::-1])
    ax[0].set_ylim(0,nvariables)
    ax[0].tick_params('both',length=0,width=0,which='major')
    if vcolor is not None:
        for i,t in enumerate(ax[0].yaxis.get_ticklabels()):
            t.set_backgroundcolor(vcolor[::-1][i])
    
    # compute and plot the variable z-scores
    mean = data.mean(axis=1)
    std  = data.std (axis=1)
    np.seterr(invalid='ignore')
    Z    = (data-mean[:,np.newaxis])/std[:,np.newaxis]
    Z    = np.ma.masked_invalid(Z)
    np.seterr(invalid='warn')
    cmap = plt.get_cmap('RdGn')
    cmap.set_bad('k',bad)
    qc   = ax[1].pcolormesh(Z[::-1],cmap=cmap,vmin=-2,vmax=2,linewidth=0)
    div  = make_axes_locatable(ax[1])
    fig.colorbar(qc,
                 ticks=(-2,-1,0,1,2),
                 format="%+d",
                 cax=div.append_axes("bottom", size="5%", pad=0.05),
                 orientation="horizontal",
                 label="Variable Z-score")
    plt.tick_params(which='both', length=0)
    ax[1].xaxis.tick_top()
    ax[1].set_xticks(np.arange(nmodels)+0.5)
    ax[1].set_xticklabels(models,rotation=90)
    ax[1].tick_params('both',length=0,width=0,which='major')
    ax[1].set_yticks([])
    ax[1].set_ylim(0,nvariables)

    # save figure
    fig.savefig(figname)

def WhittakerDiagram(X,Y,Z,**keywords):
    """Creates a Whittaker diagram.
    
    Parameters
    ----------
    X : ILAMB.Variable.Variable
       the first independent axis, classically representing temperature
    Y : ILAMB.Variable.Variable
       the second independent axis, classically representing precipitation
    Z : ILAMB.Variable.Variable
       the dependent axis
    X_plot_unit,Y_plot_unit,Z_plot_unit : str, optional
       the string representing the units of the corresponding variable
    region : str, optional
       the string representing the region overwhich to plot the diagram
    X_min,Y_min,Z_min : float, optional
       the minimum plotted value of the corresponding variable
    X_max,Y_max,Z_max : float, optional
       the maximum plotted value of the corresponding variable
    X_label,Y_label,Z_label : str, optional
       the labels of the corresponding variable
    filename : str, optional
       the output filename
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # possibly integrate in time
    if X.temporal: X = X.integrateInTime(mean=True)
    if Y.temporal: Y = Y.integrateInTime(mean=True)
    if Z.temporal: Z = Z.integrateInTime(mean=True)
    
    # convert to plot units
    X_plot_unit = keywords.get("X_plot_unit",X.unit)
    Y_plot_unit = keywords.get("Y_plot_unit",Y.unit)
    Z_plot_unit = keywords.get("Z_plot_unit",Z.unit)
    if X_plot_unit is not None: X.convert(X_plot_unit)
    if Y_plot_unit is not None: Y.convert(Y_plot_unit)
    if Z_plot_unit is not None: Z.convert(Z_plot_unit)
    
    # flatten data, if any data is masked all the data is masked
    mask   = (X.data.mask + Y.data.mask + Z.data.mask)==0

    # mask outside region
    from constants import regions as ILAMBregions
    region    = keywords.get("region","global")
    lats,lons = ILAMBregions[region]
    mask     += (np.outer((X.lat>lats[0])*(X.lat<lats[1]),
                          (X.lon>lons[0])*(X.lon<lons[1]))==0)
    x    = X.data[mask].flatten()
    y    = Y.data[mask].flatten()
    z    = Z.data[mask].flatten()

    # make plot
    fig,ax = plt.subplots(figsize=(6,5.25),tight_layout=True)
    sc     = ax.scatter(x,y,c=z,linewidths=0,
                        vmin=keywords.get("Z_min",z.min()),
                        vmax=keywords.get("Z_max",z.max()))
    div    = make_axes_locatable(ax)
    fig.colorbar(sc,cax=div.append_axes("right",size="5%",pad=0.05),
                 orientation="vertical",
                 label=keywords.get("Z_label","%s %s" % (Z.name,Z.unit)))
    X_min = keywords.get("X_min",x.min())
    X_max = keywords.get("X_max",x.max())
    Y_min = keywords.get("Y_min",y.min())
    Y_max = keywords.get("Y_max",y.max())
    ax.set_xlim(X_min,X_max)
    ax.set_ylim(Y_min,Y_max)
    ax.set_xlabel(keywords.get("X_label","%s %s" % (X.name,X.unit)))
    ax.set_ylabel(keywords.get("Y_label","%s %s" % (Y.name,Y.unit)))
    #ax.grid()
    fig.savefig(keywords.get("filename","whittaker.png"))
    plt.close()

    
