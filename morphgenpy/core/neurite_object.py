import copy

class NeuriteObject:
    """Represent one section of a rooted neurite tree."""

    def __init__(self, section_type=None, parent=None):
        """
        Parameters
        ----------
        section_type : object, optional
            Identifier describing the section type.
        parent : Neurite, optional
            Parent section.
        internal_root : bool, optional
            Treat as a root in calculating distances.
        """
        self.section_type = section_type
        self._parent = None
        self._children = []

        if parent is not None:
            self.connect(parent, relation="parent")


    @property
    def parent(self):
        """Return a copy of the child sections."""
        return self._parent

    @property
    def children(self):
        """Return a copy of the child sections."""
        return self._children.copy()


    @staticmethod
    def _connect(parent, child):
      if not (isinstance(parent, NeuriteObject) and isinstance(child, NeuriteObject)):
        raise TypeError("Neurite must be a Neurite class.")
          
      if child.parent:
        raise ValueError("Child has already a parent: disconnect from it before connecting to a new one.")
      
      if parent in child.subtree:
        raise ValueError("Parent is already a descendant of the child.")

      # make the connection
      parent._children.append(child)
      child._parent = parent
      

    @staticmethod
    def _disconnect(parent, child):
      if not child.parent:
        raise ValueError("The neurite has no parent.")
      
      if not parent.children:
        raise ValueError("The neurite has no child.")
      
      if child.parent != parent:
        raise ValueError("Child has a different parent.")
      
      if child not in parent.children:
        raise ValueError("Child is not connected to parent.")
        
      # make the connection
      parent._children.remove(child)
      child._parent = None
      
      
    @staticmethod
    def _get_relation(ref_section, target_section):
      if target_section is ref_section._parent:
        return "parent"

      if target_section is ref_section._children:
        return "child"

      raise ValueError("Unknown relation.")
    
    
    def disconnect_from_parent(self):
        """Disconnect this neurite from its parent and return the parent."""
        NeuriteObject._disconnect(self.parent, self)
        

    def disconnect_from_children(self):
        """Disconnect and return all child neurites."""
        for child in self.children:
            NeuriteObject._disconnect(self, child)
            
            
    def connect(self, neurite, relation="parent"):
        """
        Connect another neurite as this neurite's parent or child.

        Parameters
        ----------
        neurite : Neurite
            Neurite to connect.
        relation : {"parent", "child"}, default "parent"
            Relationship of ``neurite`` relative to this NeuriteObject.
        """
        if relation not in {"child", "parent"}:
          raise ValueError(f"Unknown relation {relation}")
        
        match relation:
          case "child":
            NeuriteObject._connect(self, neurite)
          case "parent":
            NeuriteObject._connect(neurite, self)
            

    def disconnect(self, neurite=None):
        """
        Disconnect this neurite's parent or one of its children.

        Parameters
        ----------
        neurite : Neurite
            Specific neurite to disconnect.
        """
        # if no neurite is specified, disconnect from both parent and children
        if neurite is None:
            self.disconnect_from_parent()
            self.disconnect_from_children()
            return

        # if it is a parent
        match NeuriteObject._get_relation(self, neurite):
          case "child":
            NeuriteObject._disconnect(self, neurite)
          case "parent":
            NeuriteObject._disconnect(neurite, self)


    @staticmethod
    def _root_and_depth(neurite):
      depth = 0

      while neurite.parent:
        depth += 1
        neurite = neurite.parent

      return {
        'parent':neurite,
        'depth':depth
        }
      
    @property
    def root(self):
        """Return the root section of the tree."""
        return NeuriteObject._root_and_depth(self)['parent']

      
    @property
    def depth(self):
        """Return the number of connections from this section to the root."""
        return NeuriteObject._root_and_depth(self)['depth']


    @staticmethod
    def _visit_tree(neurite):
      branches = [neurite]
      
      i = 0
      while i < len(branches):
        branches += branches[i].children
        i += 1
        
      return branches
      

    @property
    def subtree(self):
        """Return this section and all its descendants."""
        return NeuriteObject._visit_tree(self)


    @property
    def wholetree(self):
        """Return all sections belonging to the same tree."""
        return self.root.subtree

    @property
    def siblings(self):
        """Return sibling sections."""

        if not self.parent:
          return []
        
        siblings = self.parent.children
        siblings.remove(self)

        return siblings

      
    @property
    def bifurcation_count(self):
        """Return the number of branching sections in this subtree."""
        return sum(
            len(neurite.children) == 2
            for neurite in self.subtree if neurite.section_type != "soma"
        )


    @property
    def total_length(self):
        """Return the total Euclidean length of this subtree."""        
        return sum(neurite.length for neurite in self.subtree)


    @property
    def distance_from_root(self):
        """Return the path distance to the start of the section."""
        if self.section_type == "soma" or not self.parent:
            return 0.0

        return self.parent.distance_from_root + self.parent.length    


    @property
    def has_children(self):
      """ indicate if the neurite has at least one children """
      return len(self.children) > 0


    def clone(self):
      """ clone a class """
      return copy.deepcopy(self)

    def sholl_plot(self, bin_size, max_distance=None):
        
        """
        Calculate a Sholl plot after aligning primary dendrite origins.

        Bin zero contains the number of primary dendrites. Each primary
        dendrite subtree is translated so that its first point lies at the
        origin before shell intersections are calculated.
        """

        raise Exception("Sholl plot function not implemented!")


    @property
    def length(self):
        """Return the length of the NeuriteObject."""
        raise Exception("length property not implemented!")      



