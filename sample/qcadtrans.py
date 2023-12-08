#!/usr/bin/env python
# encoding: utf-8

'''
Specialised QCA Graph subclass, methods for parsing QCADesigner files, and
related methods
'''

from __future__ import print_function   # needed for verbose print

__author__      = 'Jake Retallick'
__copyright__   = 'MIT License'
__version__     = '2.0'
__date__        = '2017-11-06'

from collections import namedtuple, defaultdict
import bisect
import numpy as np

import re

from graph import Graph


def dget(d, key, default=None, mp=lambda x:x):
    '''Useful defaulted dict-like accessor method'''
    try:
        return mp(d[key])
    except:
        return default


# QCADesigner file parsing

# the following constants need to be used by local classes but should not
# be directly visible externally

# cell-type flags in QCACell
_CFs = {'normal':0, 'input':1, 'output':2, 'fixed':3}

# cell-type tags tags in QCADesigner files
_QCAD_CFs = {'QCAD_CELL_NORMAL': _CFs['normal'],
             'QCAD_CELL_INPUT':  _CFs['input'],
             'QCAD_CELL_OUTPUT': _CFs['output'],
             'QCAD_CELL_FIXED':  _CFs['fixed']}

# tags for different object types
_TAGS = {'design':  'TYPE:DESIGN',
         'cell':     'TYPE:QCADCell',
         'cell_obj': 'TYPE:QCADDesignObject',
         'dot':      'TYPE:CELL_DOT'}

# keys for cell values
_VALS = {'cx': 'cell_options.cxCell',
         'cy':  'cell_options.cyCell',
         'cf':  'cell_function'}

class ParserNode:
    '''Nested Node structure for QCADesigner's specific html-like format'''

    _tag_charset = '[a-zA-Z0-9_:]'
    _val_charset = '[a-zA-Z0-9_\.\-\+]'

    _rgx_in  = re.compile('\[({0}+)\]'.format(_tag_charset))      # start tag
    _rgx_out = re.compile('\[#({0}+)\]'.format(_tag_charset))     # end tag
    _rgx_val = re.compile('({0}+)=({0}+)'.format(_val_charset))   # key=val pair

    def __init__(self, fp, tag=None):
        '''Build the nested node hierarchy from the current file pointer

        inputs:
            fp  : current file pointer
            tag : tag of current Node
        '''

        self.tag = tag
        self.children = []
        self.d = {}
        self.rgxm = None

        for line in fp:
            if self._rgx_match(self._rgx_in, line, 1):
                self.children.append(ParserNode(fp, tag=self.rgxm))
            elif self._rgx_match(self._rgx_out, line, 1):
                if self.rgxm != self.tag:
                    print('tag mismatch: {0} :: {1}'.format(self.tag, self.rgxm))
                break
            else:
                m = self._rgx_val.match(line)
                if m.lastindex==2:
                    self.d[m.group(1)] = m.group(2)


    def __getitem__(self, key):
        '''Get the keyed Node value'''
        return self.d[key] if key in self.d else None

    def _rgx_match(self, rgx, s, n=0):
        '''March s with an re pattern and store the n^th group'''
        m = rgx.match(s)
        self.rgxm = m.group(n) if (m and m.lastindex >= n) else None
        return self.rgxm is not None

    def echo(self, npad=0):
        '''Recursive echo'''

        prefix = '*'*npad+' '
        def vprint(s):
            print('{0} {1}'.format(prefix, s))

        vprint('::NODE::')
        vprint('tag: {0}'.format(self.tag))
        vprint('nchildren: {0}'.format(len(self.children)))
        vprint('::fields')
        for key, val in self.d.items():
            vprint('{0} : {1}'.format(key, val))

        for child in self.children:
            child.echo(npad+1)

    def get_child(self, tag):
        '''Attempt to get the first child of the node with the given tag'''

        for child in self.children:
            if child.tag==tag:
                return child
        else:
            return None

    def extract_all_nested(self, tag):
        '''Get a list all Nodes at any depth below the node with the given
        tag. Will only find the firt instance of the tag in each branch'''

        if self.tag == tag:
            return [self,]

        nodes = []
        for child in self.children:
            nodes += child.extract_all_nested(tag)
        return nodes

    @staticmethod
    def parse(fname):
        '''Parse a QCADesigner file and return the ParserNode head. If file is
        invalid, return None

        input:
            fname   : filename of QCADesigner file
        '''

        head = None
        try:
            with open(fname, 'r') as fp:
                head = ParserNode(fp, 'root')
        except Exception as e:
            print('Failed to parse QCADesigner file with error:\n\t{0}'.format(e.message))
            head=None
        return head

# QCACircuit structure

class QCACircuit(Graph):
    '''General container for all relevant information about a QCA circuit.'''

    QCADot = namedtuple('QCADot', ['x', 'y', 'q'])

    CFs = _CFs          # make cell functions available through QCACircuit prefix
    R0 = 2              # maximum range of interaction in normalized units
    Q0 = 1.60218e-19    # elementary charge

    def __init__(self, fname=None, head=None,verbose=False):
        '''Initialise a QCACircuit from an optional QCADesigner file parser'''

        super(QCACircuit, self).__init__()

        self.vprint = print if verbose else lambda *a, **k : None

        self.cells = []         # ordered list through insertion
        self.__cellkeys = []    # sorting keys for self.cells

        # default cell value for sorting
        self.metric = lambda c: (self.nodes[c]['x'], self.nodes[c]['y'])

        self.clists = {'normal': [], 'input': [], 'output': [], 'fixed': []}

        if fname is not None:
            head = ParserNode.parse(fname)

        # extract cell data from node structure
        if isinstance(head, ParserNode):
            self._from_head(head)
            self.vprint('Circuit information extracted from circuit parser')
        else:
            self.spacing = 1.
            self.vprint('No valid parser data found')


    def add_cell(self, x, y, scale=True, cf='normal', pol=0, rot=False):
        '''Add a new cell to the QCACircuit at the given location. No check is
        made for overlap with an existing cell.

        inputs:
            x       : cell x position
            y       : cell y position

            optional arguments

            scale   : x and y should be scaled by the cell-spacing factor
            cf      : cell function
            pol     : cell polarization, if non-zero, overwrites cf to 'fixed'
            rot     : cell is rotated
        '''

        try:
            x, y = float(x), float(y)
        except:
            print('Invalid cell coordinates, must be castable to float')
            return

        if scale:
            x, y = x*self.spacing, y*self.spacing

        assert cf in _CFs, 'Invalid cell type'
        cf = _CFs[cf]
        pol = float(pol)
        if pol != 0:
            cf = _CFs['fixed']
        rot = bool(rot)

        dots = self._to_dots(x, y, rot, pol)

        self._add_cell(x, y, cf, pol, rot, dots)

    def _from_head(self, head):
        '''Load the QCACircuit from the head Node of a parsed QCADesigner file'''

        design = head.get_child(_TAGS['design'])
        assert design is not None, 'Invalid QCADesigner file format'

        cell_nodes = head.extract_all_nested(_TAGS['cell'])
        assert len(cell_nodes)>0, 'Empty QCADesigner file'

        # get cell-cell spacing
        cx, cy = [float(cell_nodes[0][_VALS[k]]) for k in ['cx', 'cy']]
        self.spacing = np.sqrt(cx*cy)

        # extract cell content
        for node in cell_nodes:
            self._add_from_node(node)

    def _add_from_node(self, node):
        '''Extract cell parameters from a ParserNode describing a QCACell and
        add the cell to the QCACircuit'''

        # cell position from the cell's design object
        cell_obj = node.get_child(_TAGS['cell_obj'])
        if cell_obj is None:
            return None
        x, y = [float(cell_obj[k]) for k in ['x', 'y']]

        # cell parameters
        cf = _QCAD_CFs[node[_VALS['cf']]]

        # quantum dot locations
        dots = node.extract_all_nested(_TAGS['dot'])
        assert len(dots)==4, 'Invalid dot layout'

        dots = [self.QCADot(float(d['x']), float(d['y']), float(d['charge'])/self.Q0) for d in dots]

        pol = round((dots[0].q+dots[2].q-dots[1].q-dots[3].q)/2, 5)
        rot = len(set(round(d.x, 2) for d in dots))==3

        self._add_cell(x, y, cf, pol, rot, dots)

    def _add_cell(self, x, y, cf, pol, rot, dots):
        '''handler for adding a new cell to the circuit'''

        n = len(self)
        self.add_node(n, weight=0, x=x, y=y, cf=cf, pol=pol, rot=rot, dots=dots)

        self._insort(n)

    def _to_dots(self, x, y, rot, pol):
        '''Compute suitable dots for the given QCACell parameters'''

        # dot locations
        if rot:
            dd = .5*self.spacing/np.sqrt(2)
            X = [x+dx for dx in [-dd, 0, dd, 0]]
            Y = [y+dy for dy in [0, -dd, 0, dd]]
        else:
            dd = .25*self.spacing
            X = [x+dx for dx in [dd, dd, -dd, -dd]]
            Y = [y+dy for dy in [-dd, dd, dd, -dd]]
        Q = [.5+dq for dq in [pol, -pol, pol, -pol]]

        return [self.QCADot(x, y, q) for x,y,q in zip(X,Y,Q)]

    def _insort(self, k):
        '''Insert node k into the sorted list of cells'''

        assert k in self, 'Invalid cell name'
        cell = self.nodes[k]

        key = self.metric(k)                        # sorting key
        ind = bisect.bisect(self.__cellkeys, key)   # insert location

        self.cells.insert(ind, k)
        self.__cellkeys.insert(ind, key)


if __name__ == '__main__':

    import sys

    # read file name from command line argument
    try:
        fn = sys.argv[1]
    except:
        print('missing QCADesigner file')
        sys.exit()

    # Supply the file name to the QCADesigner file and it will be parsed by the 
    # QCACircuit class.
    circuit = QCACircuit(fname=fn, verbose=False)

    # The following code outputs the following information about the cells:
    #   x and y coordinates
    #   polarization (+1 or -1 for drivers, 0 for cells)
    #   rotation (True for 45 degree rotated, False for no rotation)
    # You will have to use this information to construct your Ising Hamiltonian.
    # DO NOT use Input type cells. Fixed polarization cells are fine. Output 
    # type cells are treated the same as normal cells.
    # The nodes contain a lot of other information apart from those that are 
    # printed. Some of it could be helpful if you know how to use them, but many
    # are placeholders with the true meaning already stripped.
    # You should be able to infer inter-cell distance by the x and y 
    # displacement of adjacent cells. If you use QCADesigner in default settings,
    # the displacement should be 20.
    print('List of QCA cells from the file (drivers excluded):')
    for i in range(len(circuit.nodes)):
        node = circuit.nodes[i]
        # Here, node is a dict containing multiple key--value pairs.
        # You can use pdb to inspect what kind of information is available 
        # within each node.
        print(f'x: {node["x"]}\ty: {node["y"]}\tpol: {node["pol"]}\trotated: {node["rot"]}')
