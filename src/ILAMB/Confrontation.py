from GenericConfrontation import GenericConfrontation
import os,re
from netCDF4 import Dataset
import numpy as np

global_print_node_string  = ""
global_confrontation_list = []
global_model_list         = []

class Node(object):
    
    def __init__(self, name):
        self.name     = name
        self.children = []
        self.parent   = None
        self.weight   = None
        self.sum_weight_children = 0
        self.normalize_weight    = 0
        self.overall_weight      = 0
        self.score    = 0
        self.source   = None
        self.colormap = "jet"
        self.variable = None
        self.alternate_variable = None
        self.land     = False
        self.confrontation = None
        self.path     = None
        self.tablecolor = None
        
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
    if node.weight is not None: node.weight = float(node.weight)
    node.land = bool(node.land)
    
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
    def __init__(self,filename,regions=["global"]):
        
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
                                                          output_path=node.path,
                                                          land=node.land)
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
        
    def createHtml(self,M,filename="./_build/index.html"):

        # Create html assets
        from pylab import imsave
        def _createGradient(c1,c2):
            if c1.max()>1.: c1/=256.
            if c2.max()>1.: c2/=256.
            A = np.zeros((30,1,4))
            A[ :2,0,:] = c1
            A[-4:,0,:] = c2
            for i in range(2,26):
                a = float(i-2)/23.
                A[i,0,:] = (1.-a)*c1+a*c2
            return A
        path = "/".join(filename.split("/")[:-1]) + "/"
        image = _createGradient(np.asarray([  0.,105.,181.,256.]),
                                np.asarray([124.,184.,226.,256.]))
        imsave(path + "blue_header_bkg.png",image)
        image = _createGradient(np.asarray([256.,256.,256.,256.]),
                                np.asarray([200.,222.,239.,256.]))
        imsave(path + "blue_row_bkg.png",image)
        image = _createGradient(np.asarray([ 11.,145.,  0.,256.]),
                                np.asarray([133.,226.,124.,256.]))
        imsave(path + "green_header_bkg.png",image)
        image = _createGradient(np.asarray([256.,256.,256.,256.]),
                                np.asarray([201.,239.,200.,256.]))
        imsave(path + "green_row_bkg.png",image)         
        image = _createGradient(np.asarray([181.,  0.,  0.,256.]),
                                np.asarray([226.,124.,124.,256.]))
        imsave(path + "red_header_bkg.png",image) 
        image = _createGradient(np.asarray([256.,256.,256.,256.]),
                                np.asarray([239.,200.,200.,256.]))
        imsave(path + "red_row_bkg.png",image) 
        image = _createGradient(np.asarray([ 90., 90., 90.,256.]),
                                np.asarray([168.,168.,168.,256.]))
        imsave(path + "grey_header_bkg.png",image) 
        image = _createGradient(np.asarray([256.,256.,256.,256.]),
                                np.asarray([200.,200.,200.,256.]))
        imsave(path + "grey_row_bkg.png",image) 
        arrows = np.zeros((32,16,4))
        for i in range(7):
            arrows[ 4+i,(7-i):(7+i+1),3] = 1
            arrows[27-i,(7-i):(7+i+1),3] = 1
        imsave(path + "arrows.png",arrows)
        
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
        color:#fff;
        padding:7px 15px;
        text-align:left;
      }
      
      .child {
        color:#000;
        padding:7px 15px;
      }

      .parent {
        cursor:pointer;
      }

      .hgreen {
        background:#C9EFC8 url(green_header_bkg.png) repeat-x scroll center left;
        background-size: auto 100%;
      }
      .cgreen {
        background:#CAEEC7 none repeat-x scroll center left;
      }
      .pgreen {
        background:#fff url(green_row_bkg.png) repeat-x scroll center left;
      }

      .hblue {
        background:#7CB8E2 url(blue_header_bkg.png) repeat-x scroll center left;
        background-size: auto 100%;
      }
      .cblue {
        background:#C7DDEE none repeat-x scroll center left;
      }
      .pblue {
        background:#fff url(blue_row_bkg.png) repeat-x scroll center left;
      }

      .hred {
        background:#EFC8C8 url(red_header_bkg.png) repeat-x scroll center left;
        background-size: auto 100%;
      }
      .cred {
        background:#EFC8C8 none repeat-x scroll center left;
      }
      .pred {
        background:#fff url(red_row_bkg.png) repeat-x scroll center left;
      }

      .hgrey {
        background:#C8C8C8 url(grey_header_bkg.png) repeat-x scroll center left;
        background-size: auto 100%;
      }
      .cgrey {
        background:#C8C8C8 none repeat-x scroll center left;
      }
      .pgrey {
        background:#fff url(grey_row_bkg.png) repeat-x scroll center left;
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
                    dataset = Dataset(fname)
                    if dataset.variables.has_key("overall_score_over_global"):
                        data[ind] = dataset.variables["overall_score_over_global"][0]
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
            node.score /= sum_weights
    TraversePostorder(tree,_loadScores)


global_html = ""
global_table_color = ""

def BuildHTMLTable(tree,M):
    global global_model_list
    global_model_list = M
    def _genHTML(node):
        global global_html
        global global_table_color
        if node.isLeaf():
            weight = np.round(100.0*node.normalize_weight,1)
            if node.confrontation is None:
                global_html += """
      <tr class="child c%s">
        <td>&nbsp;&nbsp;&nbsp;%s&nbsp;(%.1f%%)</td>""" % (global_table_color,node.name,weight)
                for m in global_model_list: global_html += '\n        <td>~</td>'
            else:
                c = node.confrontation
                global_html += """
      <tr class="child c%s">
        <td>&nbsp;&nbsp;&nbsp;<a href="%s/%s.html">%s</a>&nbsp;(%.1f%%)</td>""" % (global_table_color,
                                                                                   c.output_path.replace("_build/",""),
                                                                                   c.name,c.name,weight)
                for ind in range(node.score.size):
                    global_html += '\n        <td>%.2f</td>' % (node.score[ind])  
            global_html += """
        <td></td>
      </tr>"""
        else:
            global_html += """
      <tr class="parent p%s">
        <td>%s</td>""" % (global_table_color,node.name)
            for ind,m in enumerate(global_model_list):
                try:
                    global_html += '\n        <td>%.2f</td>' % (node.score[ind])
                except:
                    global_html += '\n        <td>~</td>' 
            global_html += """
        <td><div class="arrow"></div></td>
      </tr>"""
    TraversePreorder(tree,_genHTML)
    
def GenerateTable(tree,M):
    global global_html
    global global_model_list
    global global_table_color
    CompositeScores(tree,M)
    global_model_list = M
    global_table_color = tree.tablecolor
    global_html = """
    <table>
      <tr class="header h%s">
        <th style="width:300px"> </th>""" % (global_table_color)
    for m in global_model_list:
        global_html += '\n        <th style="width:80px">%s</th>' % m.name
    global_html += """
        <th style="width:20px"></th>
      </tr>"""
    for cat in tree.children: BuildHTMLTable(cat,M)
    global_html += """
    </table>"""
    return global_html


