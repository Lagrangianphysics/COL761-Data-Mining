import timeit
from dataset_to_fsg import convert_to_fsg
from dataset_to_gspan import convert_to_gspan
from dataset_to_gaston import  convert_to_gaston
import argparse
import os
import matplotlib.pyplot as plt
# import shutil

threshold = ["95", "50", "25", "10", "5"]
TIMEOUT = 3600
total_graphs = 0

def get_code_to_run_gSpan(thres, gspan_executable, dataset):
    if thres == "5":
        thres = "05"
    thres = "0." + thres

    return f'''
try:
    res = subprocess.run(["{gspan_executable}", "-f", "{dataset}", "-s", "{thres}", "-o"], 
                         capture_output=True, text=True, check=True, timeout={TIMEOUT})
except:
    with open("{dataset}.fp" , "w") as file:
        file.write("")
    print("TLE/MLE")
'''

def get_code_to_run_FSG(thres, fsg_executable, dataset):
    thres = thres + ".0"

    return f'''
try:
    res = subprocess.run(["{fsg_executable}", "-s", "{thres}", "{dataset}"], 
                         capture_output=True, text=True, check=True, timeout={TIMEOUT})
except:
    with open("{dataset}.fp" , "w") as file:
        file.write("")
    print("TLE/MLE")
'''

def get_code_to_run_Gaston(thres, gaston_executable, dataset , opfolder):
    support = ((int(thres) * total_graphs) + 99) // 100

    return f'''
try:
    subprocess.run(["{gaston_executable}", "{support}", "{dataset}", "{opfolder}/gaston{thres}"], 
                   capture_output=True, text=True, check=True, timeout={TIMEOUT})
except:
    with open("{opfolder}/gaston{thres}" , "w") as file:
        file.write("")
    print("TLE/MLE")
'''



def main():
    global total_graphs
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsg", required=True, help="Path to FSG executable")
    parser.add_argument("--gspan", required=True, help="Path to GSPAN executable")
    parser.add_argument("--gaston", required=True, help="Path to GASTON executable")
    parser.add_argument("--dataset", required=True, help="Path to graph dataset file")
    parser.add_argument("--opfolder", required=True, help="Path to output folder")
    args = parser.parse_args()

    os.makedirs(args.opfolder, exist_ok=True)
    threshold_values = [95, 50, 25, 10, 5]
    
    # Convert the dataset into a compatible format
    print(f"---- Converting Dataset for FSG ----")
    converted_dataset_fsg_path = "dataset_fsg"
    convert_to_fsg(args.dataset, converted_dataset_fsg_path)
    print(f"---- Converting Dataset for gspan ----")
    converted_dataset_gspan_path = "dataset_gspan"
    convert_to_gspan(args.dataset, converted_dataset_gspan_path)
    print(f"---- Converting Dataset for gaston ----")
    converted_dataset_gaston_path = "dataset_gaston"
    total_graphs = convert_to_gaston(args.dataset, converted_dataset_gaston_path)

    time_fsg = []
    time_gspan = []
    time_gaston = []

    for thres in threshold:
        print(f"---- started FSG with threshold = {thres}% ----")
        code_fsg = get_code_to_run_FSG(thres, args.fsg, converted_dataset_fsg_path)
        time_taken_fsg = timeit.timeit(code_fsg, setup="import subprocess", number=1)
        time_fsg.append(time_taken_fsg)
        if time_taken_fsg < TIMEOUT :
            with open(f"{converted_dataset_fsg_path}.fp", "r") as source, open(f"{args.opfolder}/fsg{thres}", "w") as destination:
                destination.write(source.read())
        else :
            with open(f"{args.opfolder}/fsg{thres}", "w") as destination:
                destination.write("")
        print(f"Time taken = [{time_taken_fsg}]")

        print(f"---- started GSPAN with threshold = {thres}% ----")
        code_gspan = get_code_to_run_gSpan(thres, args.gspan, converted_dataset_gspan_path)
        time_taken_gspan = timeit.timeit(code_gspan, setup="import subprocess", number=1)
        time_gspan.append(time_taken_gspan)
        if time_taken_gspan < TIMEOUT :
            with open(f"{converted_dataset_gspan_path}.fp", "r") as source, open(f"{args.opfolder}/gspan{thres}", "w") as destination:
                destination.write(source.read())
        else :
            with open(f"{args.opfolder}/gspan{thres}", "w") as destination:
                destination.write("")
        print(f"Time taken = [{time_taken_gspan}]")

        print(f"---- started GASTON with threshold = {thres}% ----")
        code_gaston = get_code_to_run_Gaston(thres, args.gaston, converted_dataset_gaston_path, args.opfolder)
        time_taken_gaston = timeit.timeit(code_gaston, setup="import subprocess", number=1)
        time_gaston.append(time_taken_gaston)
        if time_taken_gaston >= TIMEOUT :
            with open(f"{args.opfolder}/gaston{thres}", "w") as destination:
                destination.write("")
        print(f"Time taken = [{time_taken_gaston}]")

    # Save results to file
    # with open(os.path.join(args.opfolder, "TIME_TAKEN.txt"), 'w') as file:
    #     file.write(f"time_fsg = {time_fsg}\ntime_gspan = {time_gspan}\ntime_gaston = {time_gaston}\n")

    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(threshold_values, time_fsg, marker='o', label='FSG')
    plt.plot(threshold_values, time_gspan, marker='s', label='GSPAN')
    plt.plot(threshold_values, time_gaston, marker='^', label='GASTON')
    plt.xlabel('Threshold (%)')
    plt.ylabel('Time Taken (seconds)')
    plt.title('Execution Time Comparison: FSG vs GSPAN vs GASTON')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.opfolder, "plot.png"))
    # plt.show()
    # for filename in os.listdir():
    #     if filename.startswith("fsg") or filename.startswith("gspan") or filename.startswith("gaston"):
    #         src_path = os.path.join(os.getcwd(), filename)
    #         dest_path = os.path.join(args.opfolder, filename)
    #         shutil.move(src_path, dest_path)

    # print(f"All FSG, GSPAN, and GASTON output files moved to {args.opfolder}")
if __name__ == "__main__":
    main()