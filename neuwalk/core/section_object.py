import copy

TYPE_LABELS = {
    0: "unknown",
    1: "soma",
    2: "axon",
    3: "basal_dendrite",
    4: "apical_dendrite",
    5: "apical_oblique",
    6: "apical_secondary_dendrite",
    7: "apical_secondary_oblique",
}

TYPE_CODES = {
    label: code
    for code, label in TYPE_LABELS.items()
}

class SectionObject:
    """Represent one section of a rooted tree."""

    def __init__(self, label=None, parent=None):
        """
        Parameters
        ----------
        _category : object, optional
            Identifier describing the label.
        parent : Section, optional
            Parent section.
        internal_root : bool, optional
            Treat as a root in calculating distances.
        """
        self.label = label

        
        
        self._parent = None
        self._children = []
        self.is_root_like = False

        if parent is not None:
            self.connect(parent, relation="parent")

    @property
    def label(self):
        """Return a copy of the child sections."""
        return self._category
    
    @label.setter
    def label(self, value):
        """Return a copy of the child sections."""
        assert type(value) == str, "Label should be described as a string"
        value = value.lower()
        assert value in TYPE_CODES.keys(), f"Unknown label {value}."
        self._category = value
    
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
      if not (isinstance(parent, SectionObject) and isinstance(child, SectionObject)):
        raise TypeError("Section must be a Section class.")
          
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
        raise ValueError("The section has no parent.")
      
      if not parent.children:
        raise ValueError("The section has no child.")
      
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
        """Disconnect this section from its parent and return the parent."""
        SectionObject._disconnect(self.parent, self)
        

    def disconnect_from_children(self):
        """Disconnect and return all child sections."""
        for child in self.children:
            SectionObject._disconnect(self, child)
            
            
    def connect(self, section, relation="parent"):
        """
        Connect another section as this section's parent or child.

        Parameters
        ----------
        section : Section
            Section to connect.
        relation : {"parent", "child"}, default "parent"
            Relationship of ``section`` relative to this SectionObject.
        """
        if relation not in {"child", "parent"}:
          raise ValueError(f"Unknown relation {relation}")
        
        match relation:
          case "child":
            SectionObject._connect(self, section)
          case "parent":
            SectionObject._connect(section, self)
            

    def disconnect(self, section=None):
        """
        Disconnect this section's parent or one of its children.

        Parameters
        ----------
        section : Section
            Specific section to disconnect.
        """
        # if no section is specified, disconnect from both parent and children
        if section is None:
            
            if self.parent:
                self.disconnect_from_parent()
                
            if self.children:
                self.disconnect_from_children()
                
            return
        

        # if it is a parent
        match SectionObject._get_relation(self, section):
          case "child":
            SectionObject._disconnect(self, section)
          case "parent":
            SectionObject._disconnect(section, self)


    @staticmethod
    def _root_and_depth(section, consider_root_like=False):
      depth = 0

      while section.parent and not (consider_root_like and section.is_root_like):
        depth += 1
        section = section.parent

      return {
        'root':section,
        'depth':depth
        }

    
    @property
    def root(self):
        """Return the root section of the tree."""
        return SectionObject._root_and_depth(self, False)['root']

      
    @property
    def depth(self):
        """Return the number of connections from this section to the root."""
        return SectionObject._root_and_depth(self, False)['depth']


    @staticmethod
    def _visit_tree(section):
      branches = [section]
      
      i = 0
      while i < len(branches):
        branches += branches[i].children
        i += 1
        
      return branches
      

    @property
    def subtree(self):
        """Return this section and all its descendants."""
        return SectionObject._visit_tree(self)


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
            len(section.children) == 2
            for section in self.subtree if section._category != "soma"
        )


    @property
    def total_length(self):
        """Return the total Euclidean length of this subtree."""        
        return sum(section.length for section in self.subtree)


    @property
    def distance_from_root(self):
        """Return the path distance to the start of the section."""
        if self._category == "soma" or not self.parent:
            return 0.0

        return self.parent.distance_from_root + self.parent.length    


    @property
    def has_children(self):
      """ indicate if the section has at least one children """
      return len(self.children) > 0


    def clone(self):
      """ clone a class """
      return copy.deepcopy(self)

    def sholl_plot(self, bin_size, max_distance=None):
        
        """
        Calculate a Sholl plot after aligning primary section origins.

        Bin zero contains the number of primary sections. Each primary
        section subtree is translated so that its first point lies at the
        origin before shell intersections are calculated.
        """

        raise Exception("Sholl plot function not implemented!")


    @property
    def length(self):
        """Return the length of the SectionObject."""
        raise Exception("length property not implemented!")      



