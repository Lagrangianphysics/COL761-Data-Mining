import argparse
import timeit
import os
import matplotlib.pyplot as plt

def get_code_to_run_apriori(thres, ap_path, dataset, opfolder):
    return f"""
try:
    res = subprocess.run(["{ap_path}", "-s{thres}", "{dataset}", "{opfolder}/ap{thres}"], 
                         capture_output=True, text=True, check=True, timeout=3600)
except:
    with open("{opfolder}/ap{thres}", "w") as file:
        file.write("")
"""

def get_code_to_run_fpgrowth(thres, fp_path, dataset, opfolder):
    return f"""
try:
    res = subprocess.run(["{fp_path}", "-s{thres}", "{dataset}", "{opfolder}/fp{thres}"], 
                         capture_output=True, text=True, check=True, timeout=3600)
except:
    with open("{opfolder}/fp{thres}", "w") as file:
        file.write("")
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ap", required=True, help="Path to apriori executable")
    parser.add_argument("--fp", required=True, help="Path to fpgrowth executable")
    parser.add_argument("--dataset", required=True, help="Path to dataset file")
    parser.add_argument("--opfolder", required=True, help="Path to output folder")
    args = parser.parse_args()

    os.makedirs(args.opfolder, exist_ok=True)
    threshold_values = [90, 50, 25, 10, 5]
    time_apriori = []
    time_fpgrowth = []
    
    for thres in threshold_values:
        print(f"---- started apriori with threshold = {thres}% ----")
        code_ap = get_code_to_run_apriori(thres, args.ap, args.dataset, args.opfolder)
        time_taken_ap = timeit.timeit(code_ap, setup="import subprocess", number=1)
        time_apriori.append(time_taken_ap)
        print(f"Time taken = {time_taken_ap}")
        
        print(f"---- started fpgrowth with threshold = {thres}% ----")
        code_fp = get_code_to_run_fpgrowth(thres, args.fp, args.dataset, args.opfolder)
        time_taken_fp = timeit.timeit(code_fp, setup="import subprocess", number=1)
        time_fpgrowth.append(time_taken_fp)
        print(f"Time taken = {time_taken_fp}")
    
    # with open(os.path.join(args.opfolder, "TIME_TAKEN"), 'a') as file:
    #     file.write(f"time_apriori = {time_apriori}\ntime_fpgrowth = {time_fpgrowth}\n")
    
    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(threshold_values, time_apriori, marker='o', label='Apriori')
    plt.plot(threshold_values, time_fpgrowth, marker='s', label='FP-Growth')
    plt.xlabel('Threshold (%)')
    plt.ylabel('Time Taken (seconds)')
    plt.title('Execution Time Comparison: Apriori vs FP-Growth')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.opfolder, "plot.png"))
    plt.show()

if __name__ == "__main__":
    main()
