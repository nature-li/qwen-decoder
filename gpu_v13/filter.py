with open("./pd_comm.cpp") as f:
    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        if line.startswith("//"):
            continue
        if line.startswith("/*"):
            continue
        if line.startswith("*"):
            continue
        if line.startswith("*/"):
            continue
        print(line)