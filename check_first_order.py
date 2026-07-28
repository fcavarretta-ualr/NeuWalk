import argparse
from neuwalk.io.swc import read_swc, write_swc
import numpy as np

def main(tolerance=5):
    parser = argparse.ArgumentParser(
        description="Print the hierarchical structure of an SWC morphology."
    )
    parser.add_argument("swc_file", help="Path to the SWC file")
    args = parser.parse_args()

    roots = read_swc(args.swc_file)
    
##    soma = []
##    for r in roots.copy():
##        for s in r.subtree:
##            if s.section_type == "soma":
##                soma += s.points
##    soma = np.mean(soma, axis=0)
    
##    for r in roots.copy():
##        for s in r.subtree:
##            if s.parent:
##                s.points = s.points[1:]
                
    for r in roots.copy():
      for s in r.subtree:
          if s.parent and s.section_type != s.parent.section_type:
              s.disconnect_from_parent()
              s.points = s.points[1:]
              roots.append(s)
            
    for r in roots.copy():
      for s in r.subtree:
        if s.section_type == "basal_dendrite":
            if not s.parent and (len(s.points) < 2 or s.length <= tolerance):
                s.disconnect()
                continue
            
            if not s.parent and s not in roots:
                roots.append(s)
                
##    for r in roots.copy():
##      for s in r.subtree:
##          if s.parent:
##              s.points.insert(0, s.parent.points[-1])
              
    print(args.swc_file, len([r for r in roots if r.section_type == "basal_dendrite"]))
    write_swc(args.swc_file, roots)

if __name__ == "__main__":
    main()
