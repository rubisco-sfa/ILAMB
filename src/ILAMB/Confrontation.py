from GenericConfrontation import GenericConfrontation
import os,re
from netCDF4 import Dataset
import numpy as np

global_print_node_string  = ""
global_confrontation_list = []

class Node(object):
    
    def __init__(self, name):
        self.name     = name
        self.children = []
        self.parent   = None
        self.weight   = None
        self.sum_weight_children = 0
        self.normalize_weight    = 0
        self.overall_weight      = 0
        self.source   = None
        self.colormap = "jet"
        self.variable = None
        self.alternate_variable = None
        self.land     = False
        self.confrontation = None
        self.path     = None
        
    def __str__(self):
        if self.parent is None: return ""
        name   = self.name if self.name is not None else ""
        weight = self.weight
        if self.isLeaf():
            s = "%s%s %d %.2f%%" % ("   "*(self.getDepth()-1),name,weight,100*self.overall_weight)
        else:
            s = "%s%s %d" % ("   "*(self.getDepth()-1),name,weight)
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
    if node.weight is None: return
    node.weight = float(node.weight)
        
def SumWeightChildren(node):
    for child in node.children:
        if child.weight is None:
            node.sum_weight_children += child.sum_weight_children
        else:
            node.sum_weight_children += child.weight
    if node.weight is None: node.weight = node.sum_weight_children
    
def NormalizeWeights(node):
    if node.parent is not None:
        node.normalize_weight = node.weight/node.parent.sum_weight_children

def OverallWeights(node):
    if node.isLeaf():
        node.overall_weight = node.normalize_weight
        parent = node.parent
        while parent.parent is not None:
            node.overall_weight *= parent.normalize_weight
            parent = parent.parent

def ParseConfrontationConfigureFile(filename):
    root = Node(None)
    previous_node = root
    current_level = 0
    for line in file(filename).readlines():
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
            if keyword not in node.__dict__.keys(): continue
            try:
                node.__dict__[keyword] = value
            except:
                pass

    TraversePreorder (root,ConvertTypes)        
    TraversePostorder(root,SumWeightChildren)
    TraversePreorder (root,NormalizeWeights)
    TraversePreorder (root,OverallWeights)
    return root

class Confrontation():
    """
    A class for managing confrontations
    """
    def __init__(self,filename,regions=["global.large"]):
        
        self.tree = ParseConfrontationConfigureFile(filename)

        def _initConfrontation(node):
            if not node.isLeaf(): return
            try:
                node.confrontation = GenericConfrontation(node.name,
                                                          "%s/%s" % (os.environ["ILAMB_ROOT"],node.source),
                                                          node.variable,
                                                          alternate_vars=[node.alternate_variable],
                                                          regions=regions,
                                                          cmap=node.colormap,
                                                          output_path=node.path)
            except Exception,e:
                pass

        def _buildDirectories(node):
            if node.name is None: return
            path   = ""
            parent = node
            while parent.name is not None:
                path   = "%s/%s" % (parent.name.replace(" ",""),path)
                parent = parent.parent
            path = "./_build/%s" % path
            if not os.path.isdir(path): os.mkdir(path)
            node.path = path
            
        if not os.path.isdir("./_build"): os.mkdir("./_build")        
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

    def compositeScores(self,M):

        scores = {}
        
        for cat in self.confrontation.keys():
            scores[cat] = {}
            for area in self.confrontation[cat].keys():
                scores[cat][area] = {}
                for obs in self.confrontation[cat][area]:
                    data = np.zeros(len(M))
                    mask = np.ones (len(M),dtype=bool)
                    for ind,m in enumerate(M):
                        fname = "./_build/%s/%s_%s.nc" % (obs.name,obs.name,m.name)
                        if os.path.isfile(fname):
                            dset = Dataset(fname)
                            var  = [v for v in dset.variables.keys() if
                                    "rmse_score" in v and
                                    "global.large" in v]
                            if len(var) == 1:
                                data[ind] = dset.variables[var[0]][0]
                                mask[ind] = 0
                            else:
                                data[ind] = -999.
                                mask[ind] = 1
                    scores[cat][area][obs.name] = np.ma.masked_array(data,mask=mask)
        
        
    def createHtml(self,M,filename="./_build/index.html"):
        html = r"""
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html>
  <head>"""
        html += """
    <title>ILAMB Benchmark Results</title>"""
        html += """
    <script src="http://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js" type="text/javascript"></script>
    <script type="text/javascript">  
      $(document).ready(function(){
      
        function getChildren($row) {
          var children = [];
          while($row.next().hasClass('child')) {
            children.push($row.next());
            $row = $row.next();
          }            
          return children;
        }        

        $('.parent').on('click', function() {
          $(this).find(".arrow").toggleClass("up");
          var children = getChildren($(this));
          $.each(children, function() {
            $(this).toggle();
          })
        });

        $('.child').toggle();
      });
    </script>"""
        html += """
    <style>

      body {
        font-family:Arial, Helvetica, Sans-Serif;
        font-size:0.8em;
      }

      table {
        border-collapse:collapse;
      }

      div.arrow {
        background:transparent url(arrows.png) no-repeat scroll 0px -16px;
        width:16px;
        height:16px; 
        display:block;
      }
      
      div.up {
        background-position:0px 0px;
      }

      .header {
        border-collapse:collapse;
        background:#7CB8E2 url(header_bkg.png) repeat-x scroll center left;
        color:#fff;
        padding:7px 15px;
        text-align:left;
      }
      
      .child {
        background:#C7DDEE none repeat-x scroll center left;
        color:#000;
        padding:7px 15px;
      }

      .parent {
        background:#fff url(row_bkg.png) repeat-x scroll center left;
        cursor:pointer;
      }

    </style>"""
        html += """
  </head>

  <body>"""

        for tree in self.tree.children:
            html += """
    <h1>%s</h1>""" % tree.name
            html += GenerateTable(tree,M)
        html += """

</body>
</html>"""


        file(filename,"w").write(html)

def GenerateTable(tree,M):

    categories = tree.children
    
    html = """
    <table>
      <tr class="header">
        <th style="width:160px"> </th>"""
    for m in M:
        html += '\n        <th style="width:80px">%s</th>' % m.name
    html += """
        <th style="width:20px"></th>
      </tr>"""

    for cat in categories:
        html += """

      <tr class="parent">
        <td>%s</td>""" % cat.name
        for m in M:
            html += '\n        <td>1</td>'   # Actually read/compute the overall score somehow
        html += """
        <td><div class="arrow"></div></td>
      </tr>"""
        for obs in cat.children:
            html += """

      <tr class="child">
        <td>&nbsp;&nbsp;&nbsp;<a href="%s/%s.html">%s</a>&nbsp;(%.1f%%)</td>""" % (obs.path.replace("_build/",""),obs.name,obs.name,np.round(100.0*obs.normalize_weight,1))
            for m in M:
                fname = "./_build/%s/%s_%s.nc" % (obs.name,obs.name,m.name)
                score = "~"
                if os.path.isfile(fname):
                    data = Dataset(fname)
                    if "rmse_score_of_hfls_over_global.large" in data.variables.keys():
                        score = "%0.2f" % data.variables["rmse_score_of_hfls_over_global.large"][...]
                html += '\n        <td>%s</td>' % score  
            html += """
        <td></td>
      </tr>"""
                    
    html += """
    </table>"""
    return html
        
if __name__ == "__main__":
    C = Confrontation("../../demo/sample.cfg")
    
    print GenerateTable(C.tree.children[0],['a','b','c'])
