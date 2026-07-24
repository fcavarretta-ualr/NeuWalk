import argparse
from morphgenpy.io.swc import read_swc, write_swc


def main():
    parser = argparse.ArgumentParser(
        description="Print the hierarchical structure of an SWC morphology."
    )
    parser.add_argument("swc_file", help="Path to the SWC file")
    args = parser.parse_args()

    roots = read_swc(args.swc_file)
    
##    for r in roots.copy():
##        for s in r.subtree:
##            if s.parent:
##                s.points = s.points[1:]
##                
##    for r in roots.copy():
##      for s in r.subtree:
##          if s.section_type == "basal_dendrite" and s.parent and s.parent.section_type not in ["soma", "basal_dendrite"]:
##              s.disconnect()
##              roots.append(s)


            
    for r in roots.copy():
      for s in r.subtree:
        if s.section_type in ["axon", "unknown"]:
          s.disconnect()
          continue
        
        if not s.parent and s not in roots:
          roots.append(s)
          
##    for r in roots.copy():
##      for s in r.subtree:
##          if s.parent:
##              s.points.insert(0, s.parent.points[-1])
              
    write_swc(args.swc_file, roots)

if __name__ == "__main__":
    main()
