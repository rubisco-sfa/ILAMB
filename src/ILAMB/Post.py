import pylab as plt
import numpy as np
from .constants import space_opts,time_opts
from .Regions import Regions
import re
from matplotlib.colors import LinearSegmentedColormap
import os
from netCDF4 import Dataset
import pandas as pd
import json
import glob
import pickle

def UseLatexPltOptions(fsize=18):
    params = {'axes.titlesize':fsize,
              'axes.labelsize':fsize,
              'font.size':fsize,
              'legend.fontsize':fsize,
              'xtick.labelsize':fsize,
              'ytick.labelsize':fsize}
    plt.rcParams.update(params)

def UnitStringToMatplotlib(unit,add_carbon=False):
    # replace 1e-9 with nano
    match = re.findall("(1e-9\s)",unit)
    for m in match: unit = unit.replace(m,"n")
    # replace 1e-6 with micro
    match = re.findall("(1e-6\s)",unit)
    for m in match: unit = unit.replace(m,"$\mu$")
    # replace rest of 1e with 10^
    match = re.findall("1e(\-*\+*\d+)",unit)
    for m in match: unit = unit.replace("1e%s" % m,"$10^{%s}$" % m)
    # raise exponents using Latex
    tokens = unit.split()
    for token in tokens:
        old_token = token
        for m in re.findall("[a-zA-Z](\-*\+*\d)",token):
            token = token.replace(m,"$^{%s}$" % m)
        unit = unit.replace(old_token,token)
    # add carbon symbol to all mass units
    if add_carbon:
        match = re.findall("(\D*g)",unit)
        for m in match: unit = unit.replace(m,"%s C " % m)
    return unit

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
    cb = colorbar.ColorbarBase(ax,cmap=plt.get_cmap(cmap),
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

    def __init__(self,name,pattern,side=None,legend=False,benchmark=False,longname=None,width=None,br=False):

        self.name      = name
        self.pattern   = pattern
        self.side      = side
        self.legend    = legend
        self.benchmark = benchmark
        self.longname  = longname
        self.width     = width
        self.br        = br

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

        opts = "width = %d" % self.width if self.width else ""
        cls  = "break" if self.br else "container"
        code = """
        <div class="%s" id="%s_div">
          <div class="child">""" % (cls,self.name)
        if self.side is not None:
            code += """
          <center>%s</center>""" % (self.side.replace(" ","&nbsp;"))
        code += """
          <img src="" id="%s" alt="Data not available" %s></img>""" % (self.name,opts)
        if self.legend:
            code += """
          <center><img src="legend_%s.png" id="leg"  alt="Data not available" %s></img></center>""" % (self.name.replace("benchmark_",""),opts)
        code += """
          </div>
        </div>"""
        return code

def SortRegions(regions):
    if len(regions) == 0: return []
    rnames = []
    r = Regions()
    for region in regions:
        try:
            n = r.getRegionName(region)
            rnames.append(n)
        except:
            rnames.append(region)
    sorts = sorted(zip(rnames,regions))
    rnames,regions = [list(t) for t in zip(*sorts)]
    return regions
    
class HtmlPage(object):

    def __init__(self,name,title):
        self.name  = name
        self.title = title
        self.cname = ""
        self.pages = []
        self.metric_dict = None
        self.models      = None
        self.regions     = None
        self.metrics     = None
        self.units       = None
        self.priority    = ["original","Model","intersection","Benchmark","complement","Bias","RMSE","Phase","Seasonal","Spatial","Interannual","Score","Overall"]
        self.header      = "CNAME"
        self.sections    = []
        self.figures     = {}
        self.text        = None
        self.inserts     = []

    def __str__(self):

        r = Regions()
        def _sortFigures(figure):
            macro = ["timeint","timelonint","bias","rmse","iav","phase","shift","variance","spaceint","accumulate","cycle"]
            val = 1.
            for i,m in enumerate(macro):
                if m in figure.name: val += 3**i
            if figure.name.startswith("benchmark"): val -= 1.
            if figure.name.endswith("score"): val += 1.
            if figure.name.startswith("legend"):
                if "variance" in figure.name:
                    val += 1.
                else:
                    val  = 0.
            return val

        code = """
    <div data-role="page" id="%s">
      <div data-role="header" data-position="fixed" data-tap-toggle="false">
        <h1 id="%sHead">%s</h1>""" % (self.name,self.name,self.title)
        if self.pages:
            code += """
        <div data-role="navbar">
          <ul>"""
            for page in self.pages:
                opts = ""
                if page == self: opts = " class=ui-btn-active ui-state-persist"
                code += """
            <li><a href='#%s'%s>%s</a></li>""" % (page.name,opts,page.title)
            code += """
          </ul>"""
        code += """
        </div>
      </div>"""

        if self.regions:
            code += """
      <select id="%sRegion" onchange="changeRegion%s()">""" % (self.name,self.name)
            for region in self.regions:
                try:
                    rname = r.getRegionName(region)
                except:
                    rname = region
                opts  = ''
                if region == "global" or len(self.regions) == 1:
                    opts  = ' selected="selected"'
                code += """
        <option value='%s'%s>%s</option>""" % (region,opts,rname)
            code += """
      </select>"""

        if self.models:
            code += """
      <div style="display:none">
      <select id="%sModel">""" % (self.name)
            for i,model in enumerate(self.models):
                opts  = ' selected="selected"' if i == 1 else ''
                code += """
        <option value='%s'%s>%s</option>""" % (model,opts,model)
            code += """
      </select>
      </div>"""

        if self.metric_dict: code += self.metricsToHtmlTables()

        if self.text is not None:
            code += """
      %s""" % self.text

        for section in self.sections:
            if len(self.figures[section]) == 0: continue
            self.figures[section].sort(key=_sortFigures)
            code += """
        <div data-role="collapsible" data-collapsed="false"><h1>%s</h1>""" % section
            for figure in self.figures[section]:
                if figure.name == "spatial_variance": code += "<br>"
                code += "%s" % (figure)
            code += """
        </div>"""

        code += """
    </div>"""
        return code

    def setHeader(self,header):
        self.header = header

    def setSections(self,sections):
        assert type(sections) == type([])
        self.sections = sections
        for section in sections: self.figures[section] = []

    def addFigure(self,section,name,pattern,side=None,legend=False,benchmark=False,longname=None,width=None,br=False):
        assert section in self.sections
        for fig in self.figures[section]:
            if fig.name == name: return
        self.figures[section].append(HtmlFigure(name,pattern,side=side,legend=legend,benchmark=benchmark,longname=longname,width=width,br=br))

    def setMetricPriority(self,priority):
        self.priority = priority

    def metricsToHtmlTables(self):
        if not self.metric_dict: return ""
        regions = self.regions
        metrics = self.metrics
        units   = self.units
        cname   = self.cname.split(" / ")
        if len(cname) == 3:
            cname = cname[1].strip()
        else:
            cname = cname[-1].strip()
        html    = ""
        inserts = self.inserts
        j0 = 0 if "Benchmark" in self.models else -1
        score_sig = 3 # number of significant digits used in the score tables
        other_sig = 3 # number of significant digits used for non-score quantities
        for region in regions:
            html += """
        <center>
        <table class="table-header-rotated" id="%s_table_%s">
           <thead>
             <tr>
               <th></th>
               <th class="rotate"><div><span>Download Data</span></div></th>""" % (self.name,region)
            for i,metric in enumerate(metrics):
                if i in inserts: html += """
               <th></th>"""
                html += """
               <th class="rotate"><div><span>%s [%s]</span></div></th>""" % (metric,units[metric])
            html += """
             </tr>
           </thead>
           <tbody>"""

            for j,model in enumerate(self.models):
                opts = ' onclick="highlightRow%s(this)"' % (self.name) if j > j0 else ''
                html += """
             <tr>
               <td%s class="row-header">%s</td>
               <td%s><a href="%s_%s.nc" download>[-]</a></td>""" % (opts,model,opts,cname,model)
                for i,metric in enumerate(metrics):
                    sig = score_sig if "score" in metric.lower() else other_sig
                    if i in inserts: html += """
               <td%s class="divider"></td>""" % (opts)
                    add = ""
                    try:
                        tmp = self.metric_dict[model][region][metric].data
                        if tmp.mask.all():
                            add = ""
                        else:
                            add = ("%#." + "%d" % sig + "g") % tmp
                            add = add.replace("nan","")
                    except:
                        pass
                    html += """
               <td%s>%s</td>""" % (opts,add)
                html += """
             </tr>"""
            html += """
          </tbody>
        </table>
        </center>"""

        return html

    def googleScript(self):
        if not self.metric_dict: return ""
        models   = self.models
        regions  = self.regions
        metrics  = self.metrics
        units    = self.units
        cname    = self.cname.split(" / ")
        if len(cname) == 3:
            cname = cname[1].strip()
        else:
            cname = cname[-1].strip()
        rows = ""
        for section in self.sections:
            for figure in self.figures[section]:
                rows += figure.generateClickRow()

        head = """

        function updateImagesAndHeaders%s(){
            var rsel  = document.getElementById("%sRegion");
            var msel  = document.getElementById("%sModel");
            var rid   = rsel.selectedIndex;
            var mid   = msel.selectedIndex;
            var RNAME = rsel.options[rid].value;
            var MNAME = msel.options[mid].value;
            var CNAME = "%s";
            var head  = "%s";
            head      = head.replace("CNAME",CNAME).replace("RNAME",RNAME).replace("MNAME",MNAME);
            $("#%sHead").text(head);
            %s
        }""" % (self.name,self.name,self.name,self.cname,self.header,self.name,rows)

        nscores = len(metrics)
        if len(self.inserts) > 0: nscores -= self.inserts[-1]
        r0      = 2 if "Benchmark" in models else 1

        head += """

        function highlightRow%s(cell) {
            var select = document.getElementById("%sRegion");
            for (var i = 0; i < select.length; i++){
                var table = document.getElementById("%s_table_" + select.options[i].value);
                var rows  = table.getElementsByTagName("tr");
                for (var r = %d; r < rows.length; r++) {
                    for (var c = 0; c < rows[r].cells.length-%d; c++) {
                        rows[r].cells[c].style.backgroundColor = "#ffffff";
                    }
                }
                var r = cell.closest("tr").rowIndex;
                document.getElementById("%sModel").selectedIndex = r-1;
                for (var c = 0; c < rows[r].cells.length-%d; c++) {
                    rows[r].cells[c].style.backgroundColor = "#c1c1c1";
                }
            }
            updateImagesAndHeaders%s();
        }""" % (self.name,self.name,self.name,r0,nscores+1,self.name,nscores+1,self.name)

        head += """

        function paintScoreCells%s(RNAME) {

	    var PuOr = ['#b35806','#e08214','#fdb863','#fee0b6','#f7f7f7','#d8daeb','#b2abd2','#8073ac','#542788'];
	    var GnRd = ['#b2182b','#d6604d','#f4a582','#fddbc7','#f7f7f7','#d9f0d3','#a6dba0','#5aae61','#1b7837'];
	    var colors = GnRd;
            var nc = colors.length;
            var table  = document.getElementById("%s_table_" + RNAME);
            var rows   = table.getElementsByTagName("tr");
            for (var c = rows[0].cells.length-%d; c < rows[0].cells.length; c++) {
                var scores = [];
                for (var r = %d; r < rows.length; r++) {
                    val = rows[r].cells[c].innerHTML;
                    if (val=="") {
                      scores[r-%d] = 0;
                    }else{
                      scores[r-%d] = parseFloat(val);
                    }
                }
                var mean = math.mean(scores);
                var std  = math.max(0.02,math.std(scores));
                for (var r = %d; r < rows.length; r++) {
                    scores[r-%d] = Math.min(+2,Math.max(-2,(scores[r-%d]-mean)/std));
                }
                for (var r = %d; r < rows.length; r++) {
		    var e  = scores[r-%d];
                    var ae = Math.abs(e);
                    var clr;
                    if(ae >= 0.25){
                      clr = math.round(2*e+4);
                    }else{
		      clr = math.round(4*e+4);
                    }
                    clr = math.min(math.max(0,clr),8);
                    rows[r].cells[c].style.backgroundColor = colors[clr];
                }
            }
        }""" % (self.name,self.name,nscores,r0,r0,r0,r0,r0,r0,r0,r0)

        head += """

        function pageLoad%s() {
            var select = document.getElementById("%sRegion");
            var region = getQueryVariable("region");
            var model  = getQueryVariable("model");
            if (region) {
                for (var i = 0; i < select.length; i++){
                    if (select.options[i].value == region) select.selectedIndex = i;
                }
            }
            var table = document.getElementById("%s_table_" + select.options[select.selectedIndex].value);
            var rows  = table.getElementsByTagName("tr");
            if (model) {
                for (var r = 0; r < rows.length; r++) {
                    if(rows[r].cells[0].innerHTML==model) highlightRow%s(rows[r].cells[0]);
                }
            }else{
                highlightRow%s(rows[%d]);
            }
            for (var i = 0; i < select.length; i++){
                paintScoreCells%s(select.options[i].value);
            }
            changeRegion%s();
        }

        function changeRegion%s() {
            var select = document.getElementById("%sRegion");
            for (var i = 0; i < select.length; i++){
                RNAME = select.options[i].value;
                if (i == select.selectedIndex) {
                    document.getElementById("%s_table_" + RNAME).style.display = "table";
                }else{
                    document.getElementById("%s_table_" + RNAME).style.display = "none";
                }
            }
            updateImagesAndHeaders%s();
        }""" % (self.name,self.name,self.name,self.name,self.name,r0,self.name,self.name,self.name,self.name,self.name,self.name,self.name)

        return head,"pageLoad%s" % self.name,""

    def setRegions(self,regions):
        assert type(regions) == type([])
        self.regions = SortRegions(regions)

    def setMetrics(self,metric_dict):

        # Sorting function
        def _sortMetrics(name,priority=self.priority):
            val = 1.
            for i,pname in enumerate(priority):
                if pname in name: val += 2**i
            return val

        assert type(metric_dict) == type({})
        self.metric_dict = metric_dict

        # Build and sort models, regions, and metrics
        models  = list(self.metric_dict.keys())
        regions = []
        metrics = []
        units   = {}
        for model in models:
            for region in self.metric_dict[model].keys():
                if region not in regions: regions.append(region)
                for metric in self.metric_dict[model][region].keys():
                    units[metric] = self.metric_dict[model][region][metric].unit
                    if metric not in metrics: metrics.append(metric)
        models.sort(key=lambda key: key.lower())
        if "Benchmark" in models: models.insert(0,models.pop(models.index("Benchmark")))
        regions = SortRegions(regions)
        metrics.sort(key=_sortMetrics)
        self.models  = models
        if self.regions is None: self.regions = regions
        self.metrics = metrics
        self.units   = units

        tmp = [("bias" in m.lower()) for m in metrics]
        if tmp.count(True) > 0: self.inserts.append(tmp.index(True))
        tmp = [("score" in m.lower()) for m in metrics]
        if tmp.count(True) > 0: self.inserts.append(tmp.index(True))

    def head(self):
        return ""

class HtmlAllModelsPage(HtmlPage):

    def __init__(self,name,title):

        super(HtmlAllModelsPage,self).__init__(name,title)
        self.plots    = None
        self.nobench  = None
        self.nolegend = []

    def _populatePlots(self):

        self.plots   = []
        bench        = []
        for page in self.pages:
            if page.sections is not None:
                for section in page.sections:
                    if len(page.figures[section]) == 0: continue
                    for figure in page.figures[section]:
                        if (figure.name in ["spatial_variance","compcycle","profile",
                                            "legend_spatial_variance","legend_compcycle"]): continue # ignores
                        if "benchmark" in figure.name:
                            if figure.name not in bench: bench.append(figure.name)
                            continue
                        if figure not in self.plots: self.plots.append(figure)
                        if not figure.legend: self.nolegend.append(figure.name)
        self.nobench = [plot.name for plot in self.plots if "benchmark_%s" % (plot.name) not in bench]
        
    def __str__(self):

        if self.plots is None: self._populatePlots()
        r = Regions()

        code = """
    <div data-role="page" id="%s">
      <div data-role="header" data-position="fixed" data-tap-toggle="false">
        <h1 id="%sHead">%s</h1>""" % (self.name,self.name,self.title)
        if self.pages:
            code += """
        <div data-role="navbar">
          <ul>"""
            for page in self.pages:
                opts = ""
                if page == self: opts = " class=ui-btn-active ui-state-persist"
                code += """
            <li><a href='#%s'%s>%s</a></li>""" % (page.name,opts,page.title)
            code += """
          </ul>"""
        code += """
        </div>
      </div>"""

        if self.regions:
            code += """
      <select id="%sRegion" onchange="AllSelect()">""" % (self.name)
            for region in self.regions:
                try:
                    rname = r.getRegionName(region)
                except:
                    rname = region
                opts  = ''
                if region == "global" or len(self.regions) == 1:
                    opts  = ' selected="selected"'
                code += """
        <option value='%s'%s>%s</option>""" % (region,opts,rname)
            code += """
      </select>"""

        if self.plots:
            code += """
      <select id="%sPlot" onchange="AllSelect()">""" % (self.name)
            for plot in self.plots:
                name  = ''
                if plot.name in space_opts:
                    name = space_opts[plot.name]["name"]
                elif plot.name in time_opts:
                    name = time_opts[plot.name]["name"]
                elif plot.longname is not None:
                    name = plot.longname
                if "rel_" in plot.name: name = plot.name.replace("rel_","Relationship with ")
                if name == "": continue
                opts  = ''
                if plot.name == "timeint" or len(self.plots) == 1:
                    opts  = ' selected="selected"'
                code += """
        <option value='%s'%s>%s</option>""" % (plot.name,opts,name)
            code += """
      </select>"""

            fig        = self.plots[0]
            rem_side   = fig.side
            fig.side   = "MNAME"
            rem_leg    = fig.legend
            fig.legend = True
            img        = "%s" % (fig)
            img        = img.replace('"leg"','"MNAME_legend"').replace("%s" % fig.name,"MNAME")
            fig.side   = rem_side
            fig.legend = rem_leg
            if "Benchmark" not in self.pages[0].models:
                code += '<div id="Benchmark_div"></div>'
            for model in self.pages[0].models:
                code += img.replace("MNAME",model)

        if self.text is not None:
            code += """
      %s""" % self.text

        code += """
    </div>"""
        return code

    def googleScript(self):
        head = self.head()
        return head,"",""

    def head(self):

        if self.plots is None: self._populatePlots()

        models  = self.pages[0].models
        regions = self.regions
        try:
            regions.sort()
        except:
            pass
        head    = """
      function AllSelect() {
        var header = "%s";
        var CNAME  = "%s";
        header     = header.replace("CNAME",CNAME);
        var rid    = document.getElementById("%s").selectedIndex;
        var RNAME  = document.getElementById("%s").options[rid].value;
        var pid    = document.getElementById("%s").selectedIndex;
        var PNAME  = document.getElementById("%s").options[pid].value;
        header     = header.replace("RNAME",RNAME);
        $("#%sHead").text(header);""" % (self.header,self.cname,self.name+"Region",self.name+"Region",self.name+"Plot",self.name+"Plot",self.name)
        cond  = " || ".join(['PNAME == "%s"' % n for n in self.nobench])
        if cond == "": cond = "0"
        head += """
        if(%s){
          document.getElementById("Benchmark_div").style.display = 'none';
        }else{
          document.getElementById("Benchmark_div").style.display = 'inline';
        }""" % (cond)

        cond  = " || ".join(['PNAME == "%s"' % n for n in self.nolegend])
        if cond == "": cond = "0"
        head += """
        if(%s){""" % cond
        for model in models:
            head += """
          document.getElementById("%s_legend").style.display = 'none';""" % model
        head += """
        }else{"""
        for model in models:
            head += """
          document.getElementById("%s_legend").style.display = 'inline';""" % model
        head += """
        }"""
        for model in models:
            head += """
        document.getElementById('%s').src = '%s_' + RNAME + '_' + PNAME + '.png';
        document.getElementById('%s_legend').src = 'legend_' + PNAME + '.png';""" % (model,model,model)
        head += """
      }

      $(document).on('pageshow', '[data-role="page"]', function(){
        AllSelect()
      });"""
        return head

class HtmlSitePlotsPage(HtmlPage):

    def __init__(self,name,title):

        super(HtmlSitePlotsPage,self).__init__(name,title)

    def __str__(self):

        # setup page navigation
        code = """
    <div data-role="page" id="%s">
      <div data-role="header" data-position="fixed" data-tap-toggle="false">
        <h1 id="%sHead">%s</h1>""" % (self.name,self.name,self.title)
        if self.pages:
            code += """
        <div data-role="navbar">
          <ul>"""
            for page in self.pages:
                opts = ""
                if page == self: opts = " class=ui-btn-active ui-state-persist"
                code += """
            <li><a href='#%s'%s>%s</a></li>""" % (page.name,opts,page.title)
            code += """
          </ul>"""
        code += """
        </div>
      </div>"""

        code += """
      <select id="%sModel" onchange="%sMap()">""" % (self.name,self.name)
        for model in self.models:
            code += """
        <option value='%s'>%s</option>""" % (model,model)
        code += """
      </select>"""

        code += """
      <select id="%sSite" onchange="%sMap()">""" % (self.name,self.name)
        for site in self.sites:
            code += """
        <option value='%s'>%s</option>""" % (site,site)
        code += """
      </select>"""

        code += """
      <center>
        <div id='map_canvas'></div>
        <div><img src="" id="time" alt="Data not available"></img></div>
      </center>"""

        code += """
    </div>"""

        return code

    def setMetrics(self,metric_dict):
        self.models.sort()

    def googleScript(self):

        callback = "%sMap()" % (self.name)
        head = """
      function %sMap() {
        var sitedata = google.visualization.arrayToDataTable(
          [['Latitude', 'Longitude', '%s [%s]'],\n""" % (self.name,self.vname,self.unit)

        for lat,lon,val in zip(self.lat,self.lon,self.vals):
            if val is np.ma.masked:
                sval = "null"
            else:
                sval = "%.2f" % val
            head += "           [%.3f,%.3f,%s],\n" % (lat,lon,sval)
        head = head[:-2] + "]);\n"
        head += ("        var names = %s;" % (self.sites)).replace("u'","'").replace(", '",",'")
        head += """
        var options = {
          dataMode: 'markers',
          magnifyingGlass: {enable: true, zoomFactor: 3.},
        };
        var container = document.getElementById('map_canvas');
        var geomap    = new google.visualization.GeoChart(container);
        function updateMap() {
          var mid    = document.getElementById("%sModel").selectedIndex;
          var MNAME  = document.getElementById("%sModel").options[mid].value;
          var rid    = document.getElementById("%sSite" ).selectedIndex;
          var RNAME  = document.getElementById("%sSite" ).options[rid].value;
          document.getElementById('time').src = MNAME + '_' + RNAME + '_time.png';
        }
        function clickMap() {
          var select = geomap.getSelection();
          if (Object.keys(select).length == 1) {
            var site = $("select#SitePlotsSite");
            site[0].selectedIndex = select[0].row;
            site.selectmenu('refresh');
          }
          updateMap();
        }
        google.visualization.events.addListener(geomap,'select',clickMap);
        geomap.draw(sitedata, options);
        updateMap();
      };""" % (self.name,self.name,self.name,self.name)

        return head,callback,"geomap"

    def head(self):
        return ""

class HtmlLayout():

    def __init__(self,pages,cname,years=None):

        self.pages = pages
        self.cname = cname.replace("/"," / ")
        if years is not None:
            try:
                self.cname += " / %d-%d" % (years)
            except:
                pass
        for page in self.pages:
            page.pages = self.pages
            page.cname = self.cname

    def __str__(self):
        code = """<html>
  <head>"""

        code += """
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://code.jquery.com/mobile/1.4.5/jquery.mobile-1.4.5.min.css">
    <script src="https://code.jquery.com/jquery-1.11.3.min.js"></script>
    <script src="https://code.jquery.com/mobile/1.4.5/jquery.mobile-1.4.5.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjs/3.16.5/math.min.js"></script>
    <script>
        function getQueryVariable(variable) {
            var query = window.location.search.substring(1);
            var vars = query.split("&");
            for (var i=0;i<vars.length;i++) {
                var pair = vars[i].split("=");
                if(pair[0] == variable){return pair[1];}
            }
            return(false);
        }
    </script>"""

        functions = []
        callbacks = []
        packages  = []
        for page in self.pages:
            out = page.googleScript()
            if len(out) == 3:
                f,c,p = out
                if f != "": functions.append(f)
                if c != "": callbacks.append(c)
                if p != "": packages.append(p)

        code += """
    <script type='text/javascript'>
        function pageLoad() {"""
        for c in callbacks:
            code += """
           %s();""" % c
        code += """
        }
    </script>"""

        code += """
    <script type='text/javascript'>"""
        for f in functions:
            code += f
        code += """
    </script>"""

        max_height = 280 # will be related to max column header length across all pages
        code += """
    <style type="text/css">
      .container{
        display:inline;
      }
      .break{
        clear:left;
      }
      .child{
        margin-bottom:10px;
        display:inline-block;
        padding:5px;
        font-size: 20px;
        font-weight: bold;
      }
      table.table-header-rotated {
          border-collapse: collapse;
      }
      td {
          text-align: center;
          padding: 10px 5px;
          border: 1px solid #ccc;
      }
      th {
          padding: 5px 10px;
      }
      th.rotate {
          height: %dpx;
          white-space: nowrap;
      }
      th.rotate > div {
          transform: translate(10px, %dpx) rotate(-45deg);
          width: 0px;
      }
      th.rotate > div > span {
      }
      th.row-header {
          padding: 0px 10px;
          text-align: right;
      }
      td.divider {
          width: 0px;
          border: 0px solid #ccc;
          padding: 0px 0px
      }
    </style>""" % (max_height,max_height/2-5)

        code += """
  </head>
  <body onload="pageLoad()">"""

        ### loop over pages
        for page in self.pages: code += "%s" % (page)

        code += """
  </body>
</html>"""
        return code

def RegisterCustomColormaps():
    """Adds some new colormaps to matplotlib's database.
    """
    import colorsys as cs

    # score colormap
    cm = LinearSegmentedColormap.from_list("score",[[0.84765625, 0.37109375, 0.0078125 ],
                                                    [0.45703125, 0.4375    , 0.69921875],
                                                    [0.10546875, 0.6171875 , 0.46484375]])
    plt.register_cmap("score",cm)

    # bias colormap
    val = 0.8
    per = 0.2 /2
    Rd  = cs.rgb_to_hsv(1,0,0)
    Rd  = cs.hsv_to_rgb(Rd[0],Rd[1],val)
    Bl  = cs.rgb_to_hsv(0,0,1)
    Bl  = cs.hsv_to_rgb(Bl[0],Bl[1],val)
    RdBl = {'red':   ((0.0    , 0.0,   Bl[0]),
                      (0.5-per, 1.0  , 1.0  ),
                      (0.5+per, 1.0  , 1.0  ),
                      (1.0    , Rd[0], 0.0  )),
            'green': ((0.0    , 0.0,   Bl[1]),
                      (0.5-per, 1.0  , 1.0  ),
                      (0.5+per, 1.0  , 1.0  ),
                      (1.0    , Rd[1], 0.0  )),
            'blue':  ((0.0    , 0.0,   Bl[2]),
                      (0.5-per, 1.0  , 1.0  ),
                      (0.5+per, 1.0  , 1.0  ),
                      (1.0    , Rd[2], 0.0  ))}
    plt.register_cmap(cmap=LinearSegmentedColormap('bias',RdBl))

    cm = LinearSegmentedColormap.from_list("wetdry",[[0.545882,0.400392,0.176078],
                                                     [0.586667,0.440392,0.198824],
                                                     [0.627451,0.480392,0.221569],
                                                     [0.668235,0.520392,0.244314],
                                                     [0.709020,0.560392,0.267059],
                                                     [0.749804,0.600392,0.289804],
                                                     [0.790588,0.640392,0.312549],
                                                     [0.831373,0.680392,0.335294],
                                                     [0.872157,0.720392,0.358039],
                                                     [0.912941,0.760392,0.380784],
                                                     [0.921961,0.788039,0.399020],
                                                     [0.899216,0.803333,0.412745],
                                                     [0.876471,0.818627,0.426471],
                                                     [0.853725,0.833922,0.440196],
                                                     [0.830980,0.849216,0.453922],
                                                     [0.808235,0.864510,0.467647],
                                                     [0.785490,0.879804,0.481373],
                                                     [0.762745,0.895098,0.495098],
                                                     [0.740000,0.910392,0.508824],
                                                     [0.717255,0.925686,0.522549],
                                                     [0.680392,0.933333,0.549020],
                                                     [0.629412,0.933333,0.588235],
                                                     [0.578431,0.933333,0.627451],
                                                     [0.527451,0.933333,0.666667],
                                                     [0.476471,0.933333,0.705882],
                                                     [0.425490,0.933333,0.745098],
                                                     [0.374510,0.933333,0.784314],
                                                     [0.323529,0.933333,0.823529],
                                                     [0.272549,0.933333,0.862745],
                                                     [0.221569,0.933333,0.901961],
                                                     [0.188627,0.910196,0.922157],
                                                     [0.173725,0.863922,0.923333],
                                                     [0.158824,0.817647,0.924510],
                                                     [0.143922,0.771373,0.925686],
                                                     [0.129020,0.725098,0.926863],
                                                     [0.114118,0.678824,0.928039],
                                                     [0.099216,0.632549,0.929216],
                                                     [0.084314,0.586275,0.930392],
                                                     [0.069412,0.540000,0.931569],
                                                     [0.054510,0.493725,0.932745],
                                                     [0.052157,0.447255,0.922549],
                                                     [0.062353,0.400588,0.900980],
                                                     [0.072549,0.353922,0.879412],
                                                     [0.082745,0.307255,0.857843],
                                                     [0.092941,0.260588,0.836275],
                                                     [0.103137,0.213922,0.814706],
                                                     [0.113333,0.167255,0.793137],
                                                     [0.123529,0.120588,0.771569],
                                                     [0.133725,0.073922,0.750000],
                                                     [0.143922,0.027255,0.728431],
                                                     [0.143137,0.013725,0.703922],
                                                     [0.131373,0.033333,0.676471],
                                                     [0.119608,0.052941,0.649020],
                                                     [0.107843,0.072549,0.621569],
                                                     [0.096078,0.092157,0.594118],
                                                     [0.084314,0.111765,0.566667],
                                                     [0.072549,0.131373,0.539216],
                                                     [0.060784,0.150980,0.511765],
                                                     [0.049020,0.170588,0.484314],
                                                     [0.037255,0.190196,0.456863]])
    plt.register_cmap("wetdry",cm)                                                    

def HarvestScalarDatabase(build_dir,filename="scalar_database.csv"):
    csv = '"Section","Variable","Source","Model","ScalarName","AnalysisType","Region","ScalarType","Units","Data","Weight"'
    for root,subdirs,files in os.walk(build_dir):
        for fname in files:
            if not fname.endswith(".nc"): continue
            if "Benchmark" in fname: continue
            info = root.replace(build_dir,"")
            if info.startswith("/"): info = info[1:].split("/")
            category = info[0]
            varname  = info[1]
            provider = info[2]
            with Dataset(os.path.join(root,fname)) as dset:
                if dset.complete != 1: continue
                model = dset.getncattr("name")
                weight = dset.getncattr("weight")
                for g1 in dset.groups:
                    for g2 in dset.groups[g1].groups:
                        grp = dset.groups[g1].groups[g2]
                        for vname in grp.variables:
                            stype = "score" if "Score" in vname else "scalar"
                            region = vname.split()[-1]
                            var = vname.replace(" %s" % region,"")
                            v = grp.variables[vname]
                            V = v[...]
                            s = "nan" if V.mask else "%g" % V
                            csv += "\n" + ",".join(['"%s"' % v for v in (category,varname,provider,model,var,g1,region,stype,v.units,s,weight)])
    with open(os.path.join(build_dir,filename),mode="w") as f: f.write(csv)

def CreateJSON(csv_file,M=None):
    """Using the CSV scalar database, create a JSON following the CMEC standard.

    Parameters
    ----------
    csv_file : str
        the full path to the scalar database CSV file
    M : list of ILAMB.ModelResult, optional
        if not given, then the routine will attempt to load pickle
        files. If no models are given and they cannot be found in
        pickle files, then no description or source will be provided.

    """
    def _unCamelCase(s): return re.sub("([a-z])([A-Z])","\g<1> \g<2>",s)
    def _weightedMean(x): return (x.Data*x.Weight/x.Weight.sum()).sum()
    def _meanScore(df_local,short,*args):
        cols = ['Section','Variable','Source']
        q = df_local.query(" & ".join(["(%s == '%s')" % (col,arg) for arg,col in zip(args,cols)]))
        scores = {}
        for s in short:
            qs = q.query("ScalarName == '%s'" % s)
            if qs.shape[0] > 0: scores[_unCamelCase(s)] = _weightedMean(qs)
        return scores
    def _parseConfig(f):
        lines = open(f).readlines()
        cfg = {}
        rel = {}
        h1 = h2 = v = None
        for line in lines:
            line = line.strip()
            if line.startswith("[h1:"):
                h1 = line.strip("[h1:").strip("]").strip()
            elif line.startswith("[h2:"):
                h2 = line.strip("[h2:").strip("]").strip()
            elif line.startswith("["):
                v = line.strip("[").strip("]").strip()
                if h1 not in cfg    : cfg[h1]     = {}
                if h2 not in cfg[h1]: cfg[h1][h2] = []
                cfg[h1][h2].append(v)
            elif line.startswith("relationships"):
                line = line.replace('"','').replace("'","")
                line = (line.split("=")[1]).strip()
                line = line.split(",")
                r2 = "%s/%s" % (h2,v)
                if h2 not in rel: rel[r2] = []
                for ind in line:
                    ind = ind.strip()
                    rel[r2].append(_unCamelCase(ind))
        if rel: cfg["Relationships"] = rel
        return cfg
    
    # Drop nan's and we only need the scores from the database
    df = pd.read_csv(csv_file).dropna().query("ScalarType=='score'")
    
    # Also drop 'Overall Score' for relationships, these mess up our
    # aggregation in this routine
    q = df.query("AnalysisType=='Relationships' & ScalarName=='Overall Score'")
    df = df.drop(q.index)
    if M is None:
        models = list(df.Model.unique())
    else:
        models = [m.name for m in M]
    r = Regions()
    regions = [n for n in df.Region.unique() if n in r.regions]

    out = {}
    # meta-data for which scheme and package have been used
    out["SCHEMA"] = {"name": "CMEC","version": "v1","package": "ILAMB"}

    # what dimensions should we find?
    out["DIMENSIONS"] = {"json_structure": ["region","model","metric","statistic"]}
    out["DIMENSIONS"]["dimensions"] = {}

    # populate the regions
    nest = {}
    for region in regions:
        name = r.getRegionName(region)
        source = r.getRegionSource(region)
        nest[region] = {"LongName":name,"Description":name,"Generator":source}
    out["DIMENSIONS"]["dimensions"]["region"] = nest

    # populate the models
    if M is None:
        M = []
        for pkl_file in glob.glob(os.path.join(os.path.dirname(csv_file),"*.pkl")):
            with open(pkl_file,'rb') as infile:
                M.append(pickle.load(infile))
    nest = {}
    for model in models:
        m = [m for m in M if m.name == model]
        if len(m) > 0:
            nest[model] = {"Description":m[0].description,"Source":m[0].group}
        else:
            nest[model] = {"Description":"","Source":""}
    out["DIMENSIONS"]["dimensions"]["model"] = nest

    # populate the list of metrics
    cfg = _parseConfig(os.path.join(os.path.dirname(csv_file),"ilamb.cfg"))
    nest = {}
    base = {"URI":["https://www.osti.gov/biblio/1330803",
                   "https://doi.org/10.1029/2018MS001354"],
            "Contact": "forrest AT climatemodeling.org"}
    S = list(cfg.keys())
    for s in S:
        s_json = s
        s_csv  = s.replace(" ","")
        nest[s_json] = {"Name":s_json,"Abstract":"composite score"}
        nest[s_json].update(base)
        V = list(cfg[s].keys())
        for v in V:
            v_json = "%s::%s" % (s_json,v)
            v_csv  = v.replace(" ","")
            nest[v_json] = {"Name":v_json,"Abstract":"composite score"}
            nest[v_json].update(base)
            D = cfg[s][v]
            for d in D:
                d_json = "%s!!%s" % (v_json,d)
                d_csv  = d.replace(" ","")
                nest[d_json] = {"Name":d_json,"Abstract":"benchmark score"}
                nest[d_json].update(base)                
    out["DIMENSIONS"]["dimensions"]["metric"] = nest

    # populate list of statistics, sorted so the common ones appear first
    def _priority(key):
        val = 1
        found = False
        for i,word in enumerate(['overall','bias','rmse','cycle','interannual','spatial']):
            if word in key.lower():
                val *= 2**i
                found = True
        if not found: val = 2**6
        return val
    short = list(df.query("AnalysisType=='MeanState' & ScalarType=='score'").ScalarName.unique())
    short = sorted(short,key=_priority)
    index = [_unCamelCase(n) for n in short]
    out["DIMENSIONS"]["dimensions"]["statistic"] = {"indices":index,"short_names":short}

    # populate the statistics and their means
    nest = {}
    for region in regions:
        nest[region] = {}
        for m in models:
            nest[region][m] = {}
            df_m = df.query("AnalysisType=='MeanState' & Region=='%s' & Model=='%s'" % (region,m))
            for s in S:
                s_json = s
                s_csv  = s.replace(" ","")
                t = _meanScore(df_m,short,s_csv)
                if t: nest[region][m][s_json] = t
                V = list(cfg[s].keys())
                for v in V:
                    v_json = "%s::%s" % (s_json,v)
                    v_csv  = v.replace(" ","")
                    t = _meanScore(df_m,short,s_csv,v_csv)
                    if t: nest[region][m][v_json] = t
                    D = cfg[s][v]
                    for d in D:
                        d_json = "%s!!%s" % (v_json,d)
                        d_csv  = d.replace(" ","")
                        t = _meanScore(df_m,short,s_csv,v_csv,d_csv)
                        if t: nest[region][m][d_json] = t
                        
            df_m = df.query("AnalysisType=='Relationships' & Region=='%s' & Model=='%s'" % (region,m))
            s_json = s_csv = "Relationships"
            if len(df_m):
                nest[region][m][s_json] = {'Overall Score':_weightedMean(df_m)}
                V = list(cfg[s].keys())
                for v in V:
                    v_json = "%s::%s" % (s_json,v)
                    v_csv  = v.replace(" ","").split("/")
                    q = df_m.query("Variable=='%s' & Source=='%s'" % (v_csv[0],v_csv[1]))
                    nest[region][m][v_json] = {'Overall Score':_weightedMean(q)}
                    D = cfg[s][v]
                    for d in D:
                        d_json = "%s!!%s" % (v_json,d)
                        d_csv  = d.replace(" ","").replace("/","|")
                        q = df_m.query("Variable=='%s' & Source=='%s' & ScalarName=='%s Score'" % (v_csv[0],v_csv[1],d_csv))
                        nest[region][m][d_json] = {'Overall Score':_weightedMean(q)}
    out["RESULTS"] = nest

    with open(csv_file.replace(".csv",".json"), 'w') as outfile:
        json.dump(out,outfile)
