from .data import FlightSet

def get_last_sub_ver():
    import subprocess

    cmd = r"mc ls --recursive opensky/prc-2025-resourceful-quiver/ | grep '\.parquet$' | grep -oP '(?<=_v)\d+' | sort -n | uniq | tail -n 1"
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)

    result = proc.stdout.strip()
    if result:
        print("last version:", result)
        return int(result)
    else:
        print("no version found; stderr:", proc.stderr.strip())
        return None
    
def submit(parquet_path):
    import subprocess
    
    cmd = f"mc cp {parquet_path} opensky/prc-2025-resourcefuel-quiver/"
    proc = subprocess.run(cmd, shell=True, timeout=60)

def check_score(version):
    import subprocess
    cmd = f"mc cat opensky/prc-2025-resourceful-quiver/resourceful-quiver_v{version}.parquet_result.json"
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    result = proc.stdout.strip()
    if result:
        print(result)
    else:
        print("no version found; stderr:", proc.stderr.strip())