import json, time, urllib.request, socket
socket.setdefaulttimeout(40)

def get_numpy_url():
    with urllib.request.urlopen("https://pypi.org/pypi/numpy/1.26.4/json", timeout=30) as r:
        j = json.load(r)
    for f in j["urls"]:
        if f["filename"].endswith("cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl") and "win" not in f["filename"]:
            return f["url"]
    return None

url = get_numpy_url()
print("wheel url:", url)
path = url.split("files.pythonhosted.org", 1)[1]
hosts = {
    "pypi": "https://files.pythonhosted.org" + path,
    "tuna": "https://pypi.tuna.tsinghua.edu.cn" + path,
    "aliyun": "https://mirrors.aliyun.com/pypi/packages" + path.split("/packages",1)[1],
    "sjtu": "https://mirror.sjtu.edu.cn/pypi/web/packages" + path.split("/packages",1)[1],
}
for name, u in hosts.items():
    try:
        t = time.time()
        with urllib.request.urlopen(u) as r_:
            data = r_.read()
        mb = len(data)/1e6; dt = max(1e-6, time.time()-t)
        print(f"RESULT {name}: {round(mb,1)}MB {round(dt,1)}s {round(mb/dt,2)}MB/s")
    except Exception as e:
        print(f"RESULT {name}: FAIL {str(e)[:70]}")
