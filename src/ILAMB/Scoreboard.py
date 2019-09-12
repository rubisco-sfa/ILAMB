from .Confrontation import Confrontation
from .ConfNBP import ConfNBP
from .ConfTWSA import ConfTWSA
from .ConfRunoff import ConfRunoff
from .ConfEvapFraction import ConfEvapFraction
from .ConfIOMB import ConfIOMB
from .ConfDiurnal import ConfDiurnal
from .ConfPermafrost import ConfPermafrost
from .ConfAlbedo import ConfAlbedo
from .ConfSWE import ConfSWE
from .ConfCO2 import ConfCO2
from .ConfSoilCarbon import ConfSoilCarbon
from .ConfUncertainty import ConfUncertainty
from .Regions import Regions
import os,re
from netCDF4 import Dataset
import numpy as np
from .Post import BenchmarkSummaryFigure
from .ilamblib import MisplacedData
import glob,json

global_print_node_string  = ""
global_confrontation_list = []
global_model_list         = []

class Node(object):

    def __init__(self, name):
        self.name                = name
        self.children            = []
        self.parent              = None
        self.source              = None
        self.cmap                = None
        self.variable            = None
        self.alternate_vars      = None
        self.derived             = None
        self.land                = False
        self.confrontation       = None
        self.output_path         = None
        self.bgcolor             = "#EDEDED"
        self.table_unit          = None
        self.plot_unit           = None
        self.space_mean          = True
        self.relationships       = None
        self.ctype               = None
        self.regions             = None
        self.skip_rmse           = False
        self.skip_iav            = False
        self.mass_weighting      = False
        self.weight              = 1    # if a dataset has no weight specified, it is implicitly 1
        self.sum_weight_children = 0    # what is the sum of the weights of my children?
        self.normalize_weight    = 0    # my weight relative to my siblings
        self.overall_weight      = 0    # the multiplication my normalized weight by all my parents' normalized weights
        self.score               = 0    # placeholder

    def __str__(self):
        if self.parent is None: return ""
        name   = self.name if self.name is not None else ""
        weight = self.weight
        if self.isLeaf():
            s = "%s%s %s" % ("   "*(self.getDepth()-1),name,self.score)
        else:
            s = "%s%s %s" % ("   "*(self.getDepth()-1),name,self.score)
        return s

    def isLeaf(self):
        if len(self.children) == 0: return True
        return False

    def addChild(self, node):
        node.parent = self
        self.children.append(node)

    def getDepth(self):
        depth  = 0
        parent = self.parent
        while parent is not None:
            depth += 1
            parent = parent.parent
        return depth

def TraversePostorder(node,visit):
    for child in node.children: TraversePostorder(child,visit)
    visit(node)

def TraversePreorder(node,visit):
    visit(node)
    for child in node.children: TraversePreorder(child,visit)

def PrintNode(node):
    global global_print_node_string
    global_print_node_string += "%s\n" % (node)

def ConvertTypes(node):
    def _to_bool(a):
        if type(a) is type(True): return a
        if type(a) is type("")  : return a.lower() == "true"
    node.weight     = float(node.weight)
    node.land       = _to_bool(node.land)
    node.space_mean = _to_bool(node.space_mean)
    if node.regions        is not None: node.regions        = node.regions.split(",")
    if node.relationships  is not None: node.relationships  = node.relationships.split(",")
    if node.alternate_vars is not None:
        node.alternate_vars = node.alternate_vars.split(",")
    else:
        node.alternate_vars = []

def SumWeightChildren(node):
    for child in node.children: node.sum_weight_children += child.weight

def NormalizeWeights(node):
    if node.parent is not None:
        sumw = 1.
        if node.parent.sum_weight_children > 0: sumw = node.parent.sum_weight_children
        node.normalize_weight = node.weight/sumw

def OverallWeights(node):
    if node.isLeaf():
        node.overall_weight = node.normalize_weight
        parent = node.parent
        while parent.parent is not None:
            node.overall_weight *= parent.normalize_weight
            parent = parent.parent

def InheritVariableNames(node):
    if node.parent             is None: return
    if node.variable           is None:  node.variable       = node.parent.variable
    if node.derived            is None:  node.derived        = node.parent.derived
    if node.cmap               is None:  node.cmap           = node.parent.cmap
    if node.ctype              is None:  node.ctype          = node.parent.ctype
    if node.skip_rmse          is False: node.skip_rmse      = node.parent.skip_rmse
    if node.skip_iav           is False: node.skip_iav       = node.parent.skip_iav
    if node.mass_weighting     is False: node.mass_weighting = node.parent.mass_weighting
    node.alternate_vars = node.parent.alternate_vars

def ParseScoreboardConfigureFile(filename):
    root = Node(None)
    previous_node = root
    current_level = 0
    for line in open(filename).readlines():
        line = line.strip()
        if line.startswith("#"): continue
        m1 = re.search(r"\[h(\d):\s+(.*)\]",line)
        m2 = re.search(r"\[(.*)\]",line)
        m3 = re.search(r"(.*)=(.*)",line)
        if m1:
            level = int(m1.group(1))
            assert level-current_level<=1
            name  = m1.group(2)
            node  = Node(name)
            if   level == current_level:
                previous_node.parent.addChild(node)
            elif level >  current_level:
                previous_node.addChild(node)
                current_level = level
            else:
                addto = root
                for i in range(level-1): addto = addto.children[-1]
                addto.addChild(node)
                current_level = level
            previous_node = node

        if not m1 and m2:
            node  = Node(m2.group(1))
            previous_node.addChild(node)

        if m3:
            keyword = m3.group(1).strip()
            value   = m3.group(2).strip().replace('"','')
            try:
                node.__dict__[keyword] = value
            except:
                pass

    TraversePreorder (root,ConvertTypes)
    TraversePostorder(root,SumWeightChildren)
    TraversePreorder (root,NormalizeWeights)
    TraversePreorder (root,OverallWeights)
    TraversePostorder(root,InheritVariableNames)
    return root

def getDict(node,scalars):
    if node.name is None: return {}
    n = node
    keys = []
    while n.parent is not None:
        keys.append(n.name)
        n = n.parent
    keys = keys[::-1]
    s = scalars
    for key in keys[:-1]:
        s = s[key]['children']
    return s[keys[-1]]

def BuildDictionary(node):
    global scalars
    if node.name is None: return
    n = node
    keys = []
    while n.parent is not None:
        keys.append(n.name)
        n = n.parent
    keys = keys[::-1]
    s = scalars
    for key in keys:
        if key not in s.keys():
            s[key] = {}
            s[key]['children'] = {}
        s = s[key]['children']

def BuildScalars(node):
    if node.name is None: return
    global scalars
    global models
    global global_scores
    global section
    s = getDict(node,scalars)
    if node.isLeaf():
        files = [f for f in glob.glob(os.path.join(node.output_path,"*.nc")) if "Benchmark" not in f]
        for fname in files:
            with Dataset(fname) as dset:
                if dset.name not in models: continue
                if section not in dset.groups: continue
                grp = dset.groups[section]["scalars"]
                scores = [c for c in grp.variables.keys() if "Score" in c]
                global_scores += [c for c in scores if ((c not in global_scores) and ("global" in c))]
                for c in scores:
                    if c not in s.keys():
                        s[c] = np.ma.masked_array(np.zeros(len(models)),mask=np.ones(len(models),dtype=bool))
                    s[c][models.index(dset.name)] = grp[c][...]
    else:
        scores = None
        for child in node.children:
            if scores is None: scores = [c for c in s['children'][child.name].keys() if "children" not in c]
            for c in scores:
                if c not in s.keys():
                    s[c] = np.ma.masked_array(np.zeros(len(models)),mask=np.zeros(len(models),dtype=bool))
                if c in s['children'][child.name].keys():
                    s[c] = s[c] + s['children'][child.name][c]*child.normalize_weight

def ConvertList(node):
    if node.name is None: return
    global scalars
    s = getDict(node,scalars)
    for key in s.keys():
        if key == "children": continue
        x = s[key]
        with np.errstate(under="ignore"):
            x = (x-x.mean())/(x.std().clip(0.02) if x.std() > 1e-12 else 1)
        x.data[x.mask] = -999
        s[key] = list(x.data)

def CompositeScores(tree,M):
    global global_model_list
    global_model_list = M
    def _loadScores(node):
        if node.isLeaf():
            if node.confrontation is None: return
            data = np.zeros(len(global_model_list))
            mask = np.ones (len(global_model_list),dtype=bool)
            for ind,m in enumerate(global_model_list):
                fname = "%s/%s_%s.nc" % (node.confrontation.output_path,node.confrontation.name,m.name)
                if os.path.isfile(fname):
                    try:
                        dataset = Dataset(fname)
                        grp     = dataset.groups["MeanState"].groups["scalars"]
                    except:
                        continue
                    if "Overall Score global" in grp.variables:
                        data[ind] = grp.variables["Overall Score global"][0]
                        mask[ind] = 0
                    else:
                        data[ind] = -999.
                        mask[ind] = 1
                    node.score = np.ma.masked_array(data,mask=mask)
        else:
            node.score  = 0
            sum_weights = 0
            for child in node.children:
                node.score  += child.score*child.weight
                sum_weights += child.weight
            np.seterr(over='ignore',under='ignore')
            node.score /= sum_weights
            np.seterr(over='raise',under='raise')
    TraversePostorder(tree,_loadScores)
    
ConfrontationTypes = { None              : Confrontation,
                       "ConfNBP"         : ConfNBP,
                       "ConfTWSA"        : ConfTWSA,
                       "ConfRunoff"      : ConfRunoff,
                       "ConfEvapFraction": ConfEvapFraction,
                       "ConfIOMB"        : ConfIOMB,
                       "ConfDiurnal"     : ConfDiurnal,
                       "ConfPermafrost"  : ConfPermafrost,
                       "ConfAlbedo"      : ConfAlbedo,
                       "ConfSWE"         : ConfSWE,
                       "ConfCO2"         : ConfCO2,
                       "ConfSoilCarbon"  : ConfSoilCarbon,
                       "ConfUncertainty" : ConfUncertainty}

class Scoreboard():
    """
    A class for managing confrontations
    """
    def __init__(self,filename,regions=["global"],verbose=False,master=True,build_dir="./_build",extents=None,rel_only=False,mem_per_pair=100000.,run_title="ILAMB"):

        if 'ILAMB_ROOT' not in os.environ:
            raise ValueError("You must set the environment variable 'ILAMB_ROOT'")
        self.build_dir = build_dir
        self.rel_only  = rel_only
        self.run_title = run_title
        self.regions = regions
        
        if (master and not os.path.isdir(self.build_dir)): os.mkdir(self.build_dir)

        self.tree = ParseScoreboardConfigureFile(filename)
        max_name_len = 45

        def _initConfrontation(node):
            if not node.isLeaf(): return

            # if the user hasn't set regions, use the globally defined ones
            if node.regions is None: node.regions = regions

            # pick the confrontation to use, is it a built-in confrontation?
            if node.ctype in ConfrontationTypes:
                Constructor = ConfrontationTypes[node.ctype]
            else:
                # try importing the confrontation
                conf = __import__(node.ctype)
                Constructor = conf.__dict__[node.ctype]

            try:
                if node.cmap is None: node.cmap = "jet"
                node.source = os.path.join(os.environ["ILAMB_ROOT"],node.source)
                node.mem_slab = mem_per_pair*0.5
                node.confrontation = Constructor(**(node.__dict__))
                node.confrontation.extents = extents

                if verbose and master: print(("    {0:>%d}\033[92m Initialized\033[0m" % max_name_len).format(node.confrontation.longname))

            except MisplacedData:

                if (master and verbose):
                    longname = node.output_path
                    longname = longname.replace("//","/").replace(self.build_dir,"")
                    if longname[-1] == "/": longname = longname[:-1]
                    longname = "/".join(longname.split("/")[1:])
                    print(("    {0:>%d}\033[91m MisplacedData\033[0m" % max_name_len).format(longname))

        def _buildDirectories(node):
            if node.name is None: return
            path   = ""
            parent = node
            while parent.name is not None:
                path   = os.path.join(parent.name.replace(" ",""),path)
                parent = parent.parent
            path = os.path.join(self.build_dir,path)
            if not os.path.isdir(path) and master: os.mkdir(path)
            node.output_path = path

        TraversePreorder(self.tree,_buildDirectories)
        TraversePreorder(self.tree,_initConfrontation)

    def __str__(self):
        global global_print_node_string
        global_print_node_string = ""
        TraversePreorder(self.tree,PrintNode)
        return global_print_node_string

    def list(self):
        def _hasConfrontation(node):
            global global_confrontation_list
            if node.confrontation is not None:
                global_confrontation_list.append(node.confrontation)
        global global_confrontation_list
        global_confrontation_list = []
        TraversePreorder(self.tree,_hasConfrontation)
        return global_confrontation_list

    def createJSON(self,M,filename="scalars.json"):

        global scalars
        global models
        global global_scores
        global section
        rel_tree = GenerateRelationshipTree(self,M)
        global_scores = []
        models  = [m.name for m in M]
        scalars = {}
        TraversePreorder (self.tree,BuildDictionary)
        section = "MeanState"    ; TraversePostorder(self.tree,BuildScalars)
        TraversePreorder (self.tree,ConvertList)
        check = rel_tree.children
        if len(check) > 0: check = check[0]
        if len(check.children) > 0:
            TraversePreorder(rel_tree,BuildDictionary)
            section = "Relationships"; TraversePostorder(rel_tree,BuildScalars)
            TraversePreorder(rel_tree,ConvertList)
        with open(os.path.join(self.build_dir,filename),mode='w') as f:
            f.write("data = '%s'" % (json.dumps(scalars)))
        return global_scores,rel_tree
        
    def createHtml(self,M,filename="index.html"):
        global models
        from ILAMB.generated_version import version as ilamb_version
        r = Regions()
        run_title = "ILAMB Benchmarking" if self.run_title is None else self.run_title[0] 
        models = [m.name for m in M]
        maxM = max([len(m) for m in models])
        px = int(round(maxM*6.875))
        if px % 2 == 1: px += 1
        py = int(px/2)-5
        scores,rel_tree = self.createJSON(M)
        scores = [s.replace(" global","") for s in scores if " global" in s]
        html = """
<html>
  <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://code.jquery.com/mobile/1.4.5/jquery.mobile-1.4.5.min.css">
    <script src="https://code.jquery.com/jquery-1.11.3.min.js"></script>
    <script src="https://code.jquery.com/mobile/1.4.5/jquery.mobile-1.4.5.min.js"></script>

    <script type="text/javascript" src="scalars.json"></script>
    <script type="text/javascript">


 $(document).ready(function(){
	  function getH2Children($row) {
	      var children = [];
	      while($row.next().hasClass('child_dataset')) {
		  children.push($row.next());
		  $row = $row.next();
	      }
	      return children;
	  }
	  function getH1Children($row) {
	      var children = [];
	      var turning_on = $row.next().is(":hidden");
	      while($row.next().hasClass('child_dataset') ||
		    $row.next().hasClass('child_variable')) {
		  if(turning_on){
		      if( ($row.next().is(":hidden")) &&
			  ($row.next().hasClass('child_variable'))) children.push($row.next());
		  }else{
		      if(!($row.next().is(":hidden"))) children.push($row.next());
		  }
		  $row = $row.next();
	      }
	      return children;
	  }
	  $('.parent').on('click', function() {
	      var children = getH1Children($(this));	 
	      $.each(children, function() {
		  $(this).toggle();
	      })
	  });
	  $('.child_variable').on('click', function() {
	      var children = getH2Children($(this));
	      $.each(children, function() {
		  $(this).toggle();
	      })
	  });
	  $('.child_dataset').toggle();
      });

      function pageLoad() {
	  
	  $("table").delegate('td','mouseover mouseleave', function(e) {
	      var table = document.getElementById("scoresTable");
	      if (e.type == 'mouseover') {
		  $(this).parent().addClass("hover");
		  table.rows[0].cells[$(this).index()].style.fontWeight = "bolder";
	      }
	      else {
		  $(this).parent().removeClass("hover");
		  table.rows[0].cells[$(this).index()].style.fontWeight = "normal";
	      }
	  });

	  colorTable();
      }
      
      function printRow(table,row,array,cmap) {
	  if(typeof array == 'undefined'){
	      for(var i = 1, col; col = table.rows[row].cells[i]; i++) {
		  col.style.backgroundColor = "#808080";
	      }
	      return;
	  }
	  var nc = cmap.length;
	  for(var col=0;col<array.length;col++){
	      var clr = "#808080";
	      if(array[col] > -900){
                  var ae = Math.abs(array[col]);
                  var ind;
                  if(ae>=0.25){
                     ind = Math.round(2*array[col]+4);
                  }else{
                     ind = Math.round(4*array[col]+4);
                  }
		  ind = Math.min(Math.max(ind,0),nc-1);
		  clr = cmap[ind];
	      }
	      table.rows[row].cells[col+1].style.backgroundColor = clr;
	  }
      }
      
      function colorTable() {
	  	  
	  var scalars = JSON.parse(data);	  
	  var scalar_option = document.getElementById("ScalarOption");
          var region_option = document.getElementById("RegionOption");
	  var scalar_name   = scalar_option.options[scalar_option.selectedIndex].value;
	  scalar_name  += " " + region_option.options[region_option.selectedIndex].value;
	  
	  var PuOr = ['#b35806','#e08214','#fdb863','#fee0b6','#f7f7f7','#d8daeb','#b2abd2','#8073ac','#542788'];
	  var GnRd = ['#b2182b','#d6604d','#f4a582','#fddbc7','#f7f7f7','#d9f0d3','#a6dba0','#5aae61','#1b7837'];
	  var cmap = GnRd;
	  if(document.getElementById("colorblind").checked) cmap = PuOr;

	  var row = 1;
	  var tab = "";
	  var table = document.getElementById("scoresTable");
	  for(let h1 in scalars){
	      printRow(table,row,scalars[h1][scalar_name],cmap);
	      row += 1;
	      H1 = scalars[h1]["children"]
	      for(let h2 in H1){
		  printRow(table,row,H1[h2][scalar_name],cmap);
		  row += 1;
		  H2 = H1[h2]["children"]
		  for(let v in H2){
	              var s_name = scalar_name;
                      if(h1 == "Relationships") {
                        s_name = v.substring(0,v.indexOf("/")) + " RMSE Score " + region_option.options[region_option.selectedIndex].value;
                      }else{
                      }
		      printRow(table,row,H2[v][s_name],cmap);
		      row += 1;
		  }
	      }
	  }

	  table = document.getElementById("scoresLegend");
	  row = 0;
	  for(var col=0;col<cmap.length;col++){
	      table.rows[row].cells[col].style.backgroundColor = cmap[col];
	  }
	  
      }
    </script>
    <style type="text/css">
      .parent{
      }
      .child_variable{
      }
      .child_dataset{
      }
      table.table-header-rotated {
          border-collapse: collapse;
      }
      th.rotate {
          height: %dpx;
          white-space: nowrap;
	  font-weight: normal;
      }
      th.rotate > div {
          transform: translate(10px, %dpx) rotate(-45deg);
          width: 0px;
      }
      th.rotate > div > span {
      }
      td {
	  height: 25px;
	  width: 25px;
	  border: 1px solid;
      }
      td.row-label {
	  width: 325px;
      }
      a {
	  display:block;
	  text-decoration: none;
      }
      .hover {
	  font-weight: bold;
          border: 2px solid;
      }
    </style>
    
  </head>
  <body onload="pageLoad()">

    <div data-role="page" id="MeanState">      
      <div data-role="header" data-position="fixed" data-tap-toggle="false">
        <h1>%s</h1>
      </div>

      <select id="ScalarOption" onchange="colorTable()">""" % (px,py,run_title)

        for s in scores:
            opts  = ' selected="selected"' if "Overall" in s else ''
            html += """
        <option value="%s"%s>%s</option>""" % (s,opts,s)
        html += """
      </select>
      <select id="RegionOption" onchange="colorTable()">"""


        for region in self.regions:
            try:
                rname = r.getRegionName(region)
            except:
                rname = region
            opts  = ''
            if region == "global" or len(self.regions) == 1:
                opts  = ' selected="selected"'
            html += """
          <option value='%s'%s>%s</option>""" % (region,opts,rname)        
        html += """
      </select>

      <form>
	<fieldset data-role="controlgroup" data-type="horizontal" data-mini="True">
	  <input type="checkbox" name="colorblind" id="colorblind" checked onchange="colorTable()">
	  <label for="colorblind" >Colorblind colors</label>
	</fieldset>
      </form>
      
      <center>
	<table class="table-header-rotated" id="scoresTable">
	  <thead>
            <tr>
              <th></th>"""
        
        for m in M:
            html += """
              <th class="rotate"><div>%s</div></th>""" % (m.name)
        html += """
            </tr>
	  </thead>
	  <tbody>"""


        global global_html
        global row_color
        global_html = ""
        row_color = ""
            
        def GenRowHTML(node):
            row_class = ['','parent','child_variable','child_dataset']
            global global_html
            global row_color
            global models
            global global_sb
            d = node.getDepth()
            if d == 0: return
            if d == 1: row_color = node.bgcolor
            row_header = "%s" % (";".join(["&nbsp"]*(4*(d-1))))
            if len(row_header) > 0: row_header += ";"
            row_header += node.name
            if d == 3:
                if node.parent.parent.name == "Relationships":
                    path = node.output_path.replace(global_sb.build_dir,"")
                    html = node.output_path.replace(global_sb.build_dir,"")
                    if html.endswith("/"): html = html[:-1]
                    html = html.split("/")[-1]
                    row_link = "./%s/%s.html#Relationships" % (path,html)
                    row_link = row_link.replace("//","/")
                else:
                    row_link = "./%s/%s/%s/%s.html" % (node.parent.parent.name.replace(" ",""),
                                                       node.parent.name.replace(" ",""),
                                                       node.name.replace(" ",""),
                                                       node.name.replace(" ",""))
                row_header = '<a href="%s" target="_blank">%s</a>' % (row_link,row_header)
            
            global_html += """
	    <tr class="%s" bgcolor="%s">
              <td class="row-label">%s</td>""" % (row_class[d],row_color,row_header)
            for m in models:
                if d < 3:
                    href = ''
                else:
                    path = node.output_path.replace(global_sb.build_dir,"")
                    if "/" in node.name:
                        fname = node.output_path[:-1] if node.output_path.endswith("/") else node.output_path
                        fname = fname.split("/")[-1]
                        path = os.path.join(path,"%s.html#Relationships" % (fname))                        
                    else:
                        path = os.path.join(path,"%s.html" % (node.name))
                    if path.startswith("/"): path = path[1:]
                    href = '<a href="%s?model=%s" target="_blank">&nbsp;</a>' % (path,m)

                global_html += """
              <td>%s</td>""" % href

        global global_sb
        global_sb = self
        TraversePreorder(self.tree,GenRowHTML)
        TraversePreorder( rel_tree,GenRowHTML)
        html += global_html
        html += """
	  </tbody>
	</table>

	
	<p>Relative Scale
	<table class="table-header-rotated" id="scoresLegend">
	  <tbody>
            <tr>
              <td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td>
	    </tr>
	  </tbody>
	</table>
	Worse Value&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Better Value
	<table class="table-header-rotated" id="missingLegend">
	  <tbody>
            <tr>
              <td bgcolor="#808080"></td>
	    </tr>
	  </tbody>
	</table>Missing Data or Error
      </center>

      <div data-role="footer">
        <center>ILAMB %s</center>
      </div>
    </body>
</html>""" % (ilamb_version)

        with open("%s/%s" % (self.build_dir,filename),"w") as f:
            f.write(html)
        
    def createBarCharts(self,M):
        html = GenerateBarCharts(self.tree,M)

    def dumpScores(self,M,filename):
        
        CompositeScores(self.tree,M)
        with open("%s/%s" % (self.build_dir,filename),"w") as out:
            out.write("Variables,%s\n" % (",".join([m.name for m in M])))
            for cat in self.tree.children:
                for v in cat.children:
                    try:
                        out.write("%s,%s\n" % (v.name,','.join([str(s) for s in v.score])))
                    except:
                        out.write("%s,%s\n" % (v.name,','.join(["~"]*len(M))))


def GenerateRelationshipTree(S,M):

    # Create a tree which mimics the scoreboard for relationships, but
    # we need
    #
    # root -> category -> datasets -> relationships
    #
    # instead of
    #
    # root -> category -> variable -> datasets
    #
    rel_tree = Node(None)
    h1 = None
    for cat in S.tree.children:
        if h1 is None:
            h1 = Node("Relationships")
            h1.bgcolor = "#fff2e5"
            h1.parent  = rel_tree
            h1.normalize_weight = 1.
            rel_tree.children.append(h1)
        for var in cat.children:
            for data in var.children:
                if data               is None: continue
                if data.relationships is None: continue

                # build tree
                h2 = Node(data.confrontation.longname)
                h1.children.append(h2)
                h2.parent = h1
                h2.normalize_weight = 1.
                h2.bgcolor = h1.bgcolor
                for rel in data.relationships:
                    try:
                        longname = rel.longname
                    except:
                        longname = rel
                    v = Node(longname)
                    h2.children.append(v)
                    v.parent = h2
                    v.normalize_weight = 1./len(data.relationships)
                    v.bgcolor = h2.bgcolor
                    v.output_path = data.output_path
    return rel_tree
