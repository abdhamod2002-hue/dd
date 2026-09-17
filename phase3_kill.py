import glob, os

killed = []
for d in glob.glob("/proc/[0-9]*"):
    pid = int(d.split("/")[-1])
    try:
        cl = open(d + "/cmdline", errors="ignore").read()
    except Exception:
        continue
    if "scan_regrab" in cl:
        try:
            os.kill(pid, 9)
            killed.append(pid)
        except Exception as e:
            print("skip", pid, e)
print("killed scan pids:", killed)
