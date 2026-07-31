import argparse
from neuwalk.io.swc import read_swc, write_swc


def main():
    parser = argparse.ArgumentParser(
        description="Print the hierarchical structure of an SWC morphology."
    )
    parser.add_argument("swc_file", help="Path to the SWC file")
    args = parser.parse_args()

    roots = read_swc(args.swc_file)
    


            
    for r in roots.copy():
      for s in r.subtree:
        if s.label in ["axon", "unknown"]:
          s.disconnect()
          continue
        
        if not s.parent and s not in roots:
          roots.append(s)
          
              
    write_swc(args.swc_file, roots)

if __name__ == "__main__":
    main()
