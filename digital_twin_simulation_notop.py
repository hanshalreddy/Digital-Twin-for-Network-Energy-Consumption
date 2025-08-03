import json
from collections import defaultdict
import re
import os
import csv 
import random
from datetime import datetime, timedelta
from NCLv2 import parse_oran_topology
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import re
import pandas as pd



P_PICO_0 = (48, 51)
P_MICRO_0 = (59, 110)
P_MACRO_0 = (197, 531)

K1 = 200
P_0_ru = 200
RU_CAPACITY = 500



power_profiles = {
        "Macro": {"min": 197, "max": 531},
        "Micro": {"min": 59, "max": 110},
        "Pico": {"min": 48, "max": 51}
    }

#Highway Scenario
numMacroCell = 2
MacroRU = 9
numMicroCell = 30
MicroRU = 30

RU_CAP = 500


numMacroCU = 1
numMicroCU = 3

def build_network():
    ru_nodes, du_nodes = {}, {}
    cu_id = "CU-0"
    cu_links = []

    for i in range(numMacroCell):
        du_id = f"DU-Macro-{i}"
        ru_ids = [f"RU-Macro-{i}-{j}" for j in range(MacroRU)]
        du_nodes[du_id] = ru_ids
        for ru in ru_ids:
            ru_nodes[ru] = {"model": "Macro", "du": du_id}
        cu_links.append(du_id)

    for i in range(numMicroCell):
        du_id = f"DU-Micro-{i}"
        ru_id = f"RU-Micro-{i}"
        du_nodes[du_id] = [ru_id]
        ru_nodes[ru_id] = {"model": "Micro", "du": du_id}
        cu_links.append(du_id)

    return ru_nodes, du_nodes, cu_id, cu_links


def read_traffic(file):
    timestamps, traffic_points = [], []
    with open(file) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or len(row) < 3 or not row[0].strip() or not row[1].strip() or not row[2].strip():
                continue  # skip empty or incomplete rows
            
            timestamps.append(f"{row[0]} {row[1]}")
            traffic_points.append(float(row[2]))
    return timestamps, traffic_points

def activate_rus(ru_nodes, traffic):
    total_rus = len(ru_nodes)
    total_capacity = total_rus * RU_CAPACITY
    utilization = traffic / total_capacity
    active_count = max(1, int(utilization * total_rus))
    all_rus = list(ru_nodes.keys())
    return all_rus[:active_count]

### 1.1 bytes is when we turn off all the micro RU's

def ru_power_calc(ru_nodes, active_rus, traffic):
    ru_power = {}
    # print("trafffic:", traffic)
    # print("len of active ru:", activate_rus)
    # traffic is the total bytes.
    # ru_load is bytes for each RU.
    ru_bytes = traffic / len(active_rus) if active_rus else 0
    # print("ru_bytes:", ru_bytes)
    # 2.53 is the maximum traffic bytes in csv
    ru_load =  traffic/(2.53 * 10**8)
    for ru in ru_nodes:
        if ru in active_rus:
            model = ru_nodes[ru]["model"]
            min_p, max_p = POWER_PROFILES[model]["min"], POWER_PROFILES[model]["max"]
           

            ru_power[ru] = round(min_p + (max_p - min_p) * ru_load, 2)
        else:
            ru_power[ru] = 0.0

    return ru_power


def save_ru_power_by_type_csv(filename, ru_power_log, ru_nodes, ru_type, timestamps):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        ru_ids = [ru for ru in ru_nodes if ru_nodes[ru]["model"] == ru_type]
        writer.writerow(["Timestamp"] + ru_ids)

        for i, t in enumerate(timestamps):
            row = [t] + [ru_power_log[ru][i] for ru in ru_ids]
            writer.writerow(row)

def plot_ru_power_curve(model="Macro"):
    profile = POWER_PROFILES[model]
    utilizations = list(range(0, RU_CAPACITY + 1, 25))  # 0 to 500 Mbps in steps
    powers = [round(profile["min"] + (profile["max"] - profile["min"]) * (u / RU_CAPACITY), 2) for u in utilizations]
    os.makedirs("highway_output", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(utilizations, powers, marker='o', color='blue', label=f"{model} RU")
    plt.title(f"{model} RU Power vs Utilization")
    plt.xlabel("Utilization (Mbps)")
    plt.ylabel("Power (Watts)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"highway_output/ru_power_curve_{model.lower()}.png")
    plt.close()
    print(f"✅ Plot saved as highway_output/ru_power_curve_{model.lower()}.png")

def plot_ru_power_over_time(timestamps, ru_power_log, ru_nodes, limit=5):
    os.makedirs("highway_output", exist_ok=True)
    plt.figure(figsize=(14, 6))
    for ru in list(ru_nodes.keys())[:limit]:
        plt.plot(timestamps, ru_power_log[ru], label=ru)
    plt.xlabel("Time")
    plt.ylabel("Power (W)")
    plt.title("Sample RU Power Over Time")
    plt.grid(True)
    plt.legend()
    
    tick_interval = 6
    tick_indices = list(range(0, len(timestamps), tick_interval))
    tick_labels = []
    for i in tick_indices:
        try:
            day, hour = timestamps[i].split()
            rounded_hour = round(float(hour))
            tick_labels.append(f"{day} {rounded_hour:02}")
        except:
            tick_labels.append(timestamps[i]) 

    
    plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=45)
    plt.tight_layout()
    plt.legend(fontsize="x-small")
    plt.savefig("highway_output/sample_ru_power_over_time.png")
    plt.close()
    print("✅ Sample RU power plot saved as highway_output/sample_ru_power_over_time.png")

def plot_ru_power_by_type(timestamps, ru_power_log, ru_nodes, ru_type=None, title_suffix="All RUs", filename_suffix="all"):
    os.makedirs("highway_output", exist_ok=True)
    plt.figure(figsize=(14, 6))
    plt.grid(True)

    for ru, log in ru_power_log.items():
        model = ru_nodes[ru]["model"]
        if ru_type is None or model == ru_type:
            if ru_type == "Macro":
                log = [x * numMacroCell * MacroRU for x in log]
                plt.plot(timestamps, log, label=ru)
            if ru_type == "Micro":
                log = [x * numMicroCell + MicroRU for x in log]
                plt.plot(timestamps, log, label=ru)
            plt.plot(timestamps, log, label=ru)

    plt.xlabel("Time")
    plt.ylabel("Power (W)")
    plt.title(f"RU Power Over Time - {title_suffix}")
    tick_interval = 6
    tick_indices = list(range(0, len(timestamps), tick_interval))
    tick_labels = []
    for i in tick_indices:
        try:
            day, hour = timestamps[i].split()
            rounded_hour = round(float(hour))
            tick_labels.append(f"{day} {rounded_hour:02}")
        except:
            tick_labels.append(timestamps[i])  

    
    plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=45)
    plt.tight_layout()
    plt.legend(fontsize="xx-small", ncol = 4, loc = 'upper right')
    plt.savefig(f"highway_output/ru_power_over_time_{filename_suffix}.png")
    plt.close()
    print(f"✅ Plot saved as highway_output/ru_power_over_time_{filename_suffix}.png")


POWER_PROFILES = power_profiles  # Ensure consistent naming




# Network Power Calc
#################################################

def read_comp(file):
    timestamps = []
    totals = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            timestamps.append(row[0])  # Don't convert to datetime
            totals.append(float(row[1]))
    return timestamps, totals


def save_network_power_csv(file, timestamps, total_power):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Total Power (W)"])
        for t, p in zip(timestamps, total_power):
            writer.writerow([t, round(p, 2)])


def plot_network_comparison():
    timestamps_base, power_base = read_comp("highway_output/network_power_baseline.csv")
    timestamps_es, power_es = read_comp("highway_output/network_power_es.csv")

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps_base, power_base, label="Baseline (All RUs ON)", linestyle="--", color="blue")
    plt.plot(timestamps_es, power_es, label="ES Mode (Micro OFF < 1.1)", linestyle="-", color="red")
    plt.title("Total Network Power – Baseline vs Energy Savings")
    plt.xlabel("Time")
    plt.ylabel("Total Power (W)")
    
    plt.grid(True)
    plt.legend(fontsize="small")
    tick_interval = 6
    tick_indices = list(range(0, len(timestamps_base), tick_interval))
    tick_labels = []
    for i in tick_indices:
        try:
            day, hour = timestamps_base[i].split()
            rounded_hour = round(float(hour))
            tick_labels.append(f"{day} {rounded_hour:02}")
        except Exception:
            tick_labels.append(timestamps_base[i])  # fallback if format is unexpected

    plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=45)
    plt.tight_layout()
    
    out_path = "highway_output/network_power_comparison.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Comparison plot saved to {out_path}")

def plot_total_network_power(output_dir="highway_output"):
    # Load network-level power data
    network_base = pd.read_csv(f"{output_dir}/network_power_baseline.csv")
    #network_es = pd.read_csv(f"{output_dir}/network_power_es.csv")
    timestamps = network_base["Time"].tolist()
    
    baseline_power = network_base["Total Power (W)"]
    

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, baseline_power, label="Baseline (All RUs ON)", linestyle="--", color="blue")
    #plt.plot(timestamps, es_power, label="ES Mode (Dynamic Micro RUs)", linestyle="-", color="orange")
    plt.title("Total RU Power Consumption Over Time")
    plt.xlabel("Time")
    plt.ylabel("Total Power (W)")
    tick_interval = 6
    tick_indices = list(range(0, len(timestamps), tick_interval))
    tick_labels = []
    for i in tick_indices:
        try:
            day, hour = timestamps[i].split()
            rounded_hour = round(float(hour))
            tick_labels.append(f"{day} {rounded_hour:02}")
        except:
            tick_labels.append(timestamps[i])  # fallback if format is unexpected

    
    plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/graph1_total_ru_power.png")
    plt.close()

def plot_ru_breakdown(output_dir="highway_output"):
    # Load RU-level power data
    macro_base = pd.read_csv(f"{output_dir}/macro_ru_power_baseline.csv")
    micro_base = pd.read_csv(f"{output_dir}/micro_ru_power_baseline.csv")
    micro_es = pd.read_csv(f"{output_dir}/micro_ru_power_es.csv")

    timestamps = macro_base["Timestamp"].tolist()
    macro_power = macro_base.drop(columns="Timestamp").sum(axis=1)
    micro_power_base = micro_base.drop(columns="Timestamp").sum(axis=1)
    micro_power_es = micro_es.drop(columns="Timestamp").sum(axis=1)

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, macro_power, label="Macro RUs (Always ON)", color="green")
    plt.plot(timestamps, micro_power_base, label="Micro RUs (Baseline)", linestyle="--", color="red")
    plt.plot(timestamps, micro_power_es, label="Micro RUs (ES Algorithm)", linestyle="-", color="purple")
    plt.title("Macro and Micro RU Power Consumption Over Time")
    plt.xlabel("Time")
    plt.ylabel("Total Power (W)")
    tick_interval = 6
    tick_indices = list(range(0, len(timestamps), tick_interval))
    tick_labels = []
    for i in tick_indices:
        try:
            day, hour = timestamps[i].split()
            rounded_hour = round(float(hour))
            tick_labels.append(f"{day} {rounded_hour:02}")
        except:
            tick_labels.append(timestamps[i])  

    
    plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/graph2_ru_breakdown.png")
    plt.close()


## Energy efficieny calculator

def read_csv(file):
    timestamps = []
    data = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)[1:]  
        for row in reader:
            if not row:  
                continue
            timestamps.append(row[0]) 
           
            row_data = []
            for v in row[1:]:
                val = v.split("|")[0]  
                row_data.append(float(val))
            data.append(row_data)
    return timestamps, headers, data

def calcualte_ee_per_sector(timestamps, ru_ids, ru_loads, ru_powers, sector_map):
    sector_ee = []
    for t in range(len(timestamps)):
        row = [timestamps[t]]
        for sector, rus in sector_map.items():
            rus_clean = [r.replace(" (off)", "") for r in rus]
            ru_idxs = [ru_ids.index(r) for r in rus_clean if r in ru_ids]

            load_sum = sum([ru_loads[t][i] for i in ru_idxs])
            power_sum = sum([ru_powers[t][i] for i in ru_idxs])

            ee = load_sum / power_sum if power_sum > 0 else 0.0

            row.append(round(ee, 4))
        sector_ee.append(row)
    return sector_ee, list(sector_map.keys())

def calculate_ee_system(timestamps, ru_loads, ru_powers):
    system_ee = []
    for t in range(len(timestamps)):
        ru_load = ru_loads[t]
        ru_power = sum(ru_powers[t])

       
        ee = ru_load / ru_power if ru_power > 0 else 0.0
        system_ee.append([timestamps[t], round(ee, 4)])
    return system_ee

def write_csv(path, headers, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp"] + headers)
        for row in data:
            writer.writerow([row[0]] + row[1:])

def write_system_ee(path, data):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "EE (System)"])
        for row in data:
            writer.writerow([row[0], row[1]])


def plot_ee_system_comparison(full_csv, es_csv):
    timestamps = []
    ee_full = []
    ee_es = []

    with open(full_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            timestamps.append(row[0])
            ee_full.append(float(row[1]))

    with open(es_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            ee_es.append(float(row[1]))

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, ee_full, label="System EE – Baseline", linestyle="--", color="blue")
    plt.plot(timestamps, ee_es, label="System EE – ES", linestyle="-", color="red")

    plt.title("System Energy Efficiency – Baseline vs ES")
    plt.xlabel("Time")
    plt.ylabel("EE (Mbps/W)")
    tick_interval = 6
    tick_indices = list(range(0, len(timestamps), tick_interval))
    tick_labels = []
    for i in tick_indices:
        try:
            day, hour = timestamps[i].split()
            rounded_hour = round(float(hour))
            tick_labels.append(f"{day} {rounded_hour:02}")
        except:
            tick_labels.append(timestamps[i])  

    
    plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=45)
    plt.grid(True)
    plt.grid(True)
    plt.tight_layout()
    plt.legend(fontsize="x-small")
    out_path = "highway_output/ee_system_comparison.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()




# Simulation for Shanghai highway traffic data

def main():
    ru_nodes, du_nodes, cu_id, cu_links = build_network()

    traffic_csv = "shanghai_highway_traffic_data.csv"
    timestamps, traffic_points = read_traffic(traffic_csv)

    ru_power_log = {ru: [] for ru in ru_nodes}
    ru_status_log = {ru: [] for ru in ru_nodes}
    ru_power_log_baseline = {ru: [] for ru in ru_nodes}
    ru_power_log_es = {ru: [] for ru in ru_nodes}
    total_power_baseline = []
    total_power_es = []

    for t, traffic in zip(timestamps, traffic_points):
        active_rus = []
        for ru, info in ru_nodes.items():

            if info["model"] == "Micro" and traffic <= 1.1:
                continue  
            else:
                rangeABT = 1.4 # 2.53 - 1.1 = 1.4 range above threshold
                traffic_range = traffic - 1.1
                percetageofMicroRU = traffic_range/rangeABT
                neededMicro = int(round(percetageofMicroRU * MicroRU))
            
            micro_ru_ids = [ru for ru in ru_nodes if ru_nodes[ru]["model"] == "Micro"]
            active_micro_rus = []
            if neededMicro > 0:
                step = len(micro_ru_ids) / neededMicro
                for i in range(neededMicro):
                    index = int(i * step)
                    active_micro_rus.append(micro_ru_ids[index])

           
            active_rus = [ru for ru in ru_nodes if ru_nodes[ru]["model"] == "Macro"]
            active_rus.extend(active_micro_rus)

        traffic = traffic * 10**8
        ru_power = ru_power_calc(ru_nodes, active_rus, traffic)

        for ru in ru_nodes:
            ru_power_log[ru].append(ru_power[ru])
            ru_status_log[ru].append(1 if ru in active_rus else 0)

       

        all_rus = list(ru_nodes.keys())
        
        ### Baseline: All RUs ON
        all_rus = list(ru_nodes.keys())
        ru_power_baseline = ru_power_calc(ru_nodes, all_rus, traffic)
        for ru in ru_nodes:
            ru_power_log_baseline[ru].append(ru_power_baseline[ru])
        total_power_baseline.append(sum(ru_power_baseline.values()))

        ### ES Mode: Turn OFF Micro if traffic <= 1.1
        active_rus_es = []
        active_rus_es.extend([ru for ru in ru_nodes if ru_nodes[ru]["model"] == "Macro"])

        if traffic > 1.1 * 10**8:
            rangeABT = (2.53 - 1.1) * 10**8
            traffic_range = traffic - 1.1 * 1e8
            percentageOfMicroRU = traffic_range / rangeABT
            neededMicro = int(round(percentageOfMicroRU * numMicroCell))

            micro_ru_ids = [ru for ru in ru_nodes if ru_nodes[ru]["model"] == "Micro"]
            if neededMicro > len(micro_ru_ids):
                neededMicro = len(micro_ru_ids)
            
            active_micro_rus = []
            if neededMicro > 0:
                step = len(micro_ru_ids) / neededMicro
                for i in range(neededMicro):
                    index = int(i * step)
                    active_micro_rus.append(micro_ru_ids[index])
            active_rus_es.extend(active_micro_rus)



        ru_power_es = ru_power_calc(ru_nodes, active_rus_es, traffic)
        for ru in ru_nodes:
            ru_power_log_es[ru].append(ru_power_es[ru])
        total_power_es.append(sum(ru_power_es.values()))

    save_ru_power_by_type_csv("highway_output/macro_ru_power_baseline.csv", ru_power_log_baseline, ru_nodes, "Macro", timestamps)
    save_ru_power_by_type_csv("highway_output/macro_ru_power_es.csv", ru_power_log_es, ru_nodes, "Macro", timestamps)
    save_ru_power_by_type_csv("highway_output/micro_ru_power_baseline.csv", ru_power_log_baseline, ru_nodes, "Micro", timestamps)
    save_ru_power_by_type_csv("highway_output/micro_ru_power_es.csv", ru_power_log_es, ru_nodes, "Micro", timestamps)


    # ⬇️ Call plot function here
    plot_ru_power_over_time(timestamps, ru_power_log, ru_nodes)
    plot_ru_power_by_type(timestamps, ru_power_log, ru_nodes, ru_type="Micro", title_suffix="Capacity RUs (Micro)", filename_suffix="micro")
    plot_ru_power_by_type(timestamps, ru_power_log, ru_nodes, ru_type="Macro", title_suffix="Coverage RUs (Macro)", filename_suffix="macro")
    plot_ru_power_by_type(timestamps, ru_power_log, ru_nodes, ru_type=None, title_suffix="All RUs", filename_suffix="all")


    save_network_power_csv("highway_output/network_power_baseline.csv", timestamps, total_power_baseline)
    save_network_power_csv("highway_output/network_power_es.csv", timestamps, total_power_es)
    plot_network_comparison()
    plot_total_network_power()
    plot_ru_breakdown()


    #CU Power plot


    #EE Calc
    ts_base, _, macro_powers_base = read_csv("highway_output/macro_ru_power_baseline.csv")
    _, _, micro_powers_base = read_csv("highway_output/micro_ru_power_baseline.csv")
    _, _, macro_powers_es = read_csv("highway_output/macro_ru_power_es.csv")
    _, _, micro_powers_es = read_csv("highway_output/micro_ru_power_es.csv")
    powers_base = []
    powers_es = []
    #    Combine macro + micro 
    for i in range(len(timestamps)):
        powers_base.append(macro_powers_base[i] + micro_powers_base[i])

    for i in range(len(timestamps)):
        powers_es.append(macro_powers_es[i] + micro_powers_es[i])
    


    _, traffic_points = read_traffic("shanghai_highway_traffic_data.csv")
    MAX_TRAFFIC = 2.53
    MAX_LOAD_MBPS = 250
    traffic_mbps = [(((t * 8) / MAX_TRAFFIC) * MAX_LOAD_MBPS) * 10**8 for t in traffic_points]

    # Calculate system EE for baseline and ES
    ee_system_baseline = calculate_ee_system(ts_base, traffic_mbps, powers_base)
    ee_system_es = calculate_ee_system(ts_base, traffic_mbps, powers_es)

    # Save to CSV
    write_system_ee("highway_output/ee_system_baseline.csv", ee_system_baseline)
    write_system_ee("highway_output/ee_system_es.csv", ee_system_es)

    # Use your existing plot function
    plot_ee_system_comparison("highway_output/ee_system_baseline.csv", "highway_output/ee_system_es.csv")
   




if __name__ == "__main__":
    
    main()