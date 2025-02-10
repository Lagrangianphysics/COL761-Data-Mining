import os

def convert_to_fsg(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#'):  
                outfile.write("t\n")  
                i += 1
                num_nodes = int(lines[i].strip())  
                nodes = []
                i += 1
                for _ in range(num_nodes):
                    nodes.append(lines[i].strip())
                    i += 1
                
                num_edges = int(lines[i].strip()) 
                edges = []
                i += 1
                for _ in range(num_edges):
                    parts = lines[i].strip().split()
                    src, dest, edge_label = parts[0], parts[1], parts[2]
                    edges.append((src, dest, edge_label))
                    i += 1
                i+=1
                for idx, label in enumerate(nodes):
                    outfile.write(f"v {idx} {label}\n")
                for src, dest, edge_label in edges:
                    outfile.write(f"e {src} {dest} {edge_label}\n")

if __name__ == "__main__":
    input_filename = "/home/baadalvm/HW1/Q2/Yeast/167.txt_graph"  
    output_filename = "output.fsg"
    convert_to_fsg(input_filename, output_filename)
    print(f"Conversion complete. Output saved to {output_filename}")
