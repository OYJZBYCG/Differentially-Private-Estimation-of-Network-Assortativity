"""psutil monitor, with /proc fallback for Linux environments without psutil."""
import os
import sys
from pathlib import Path
try:
    import psutil
except ImportError:
    psutil=None


def snapshot():
    if psutil is not None:
        vm=psutil.virtual_memory();parent=psutil.Process();rss=0
        for p in [parent]+parent.children(recursive=True):
            try: rss+=p.memory_info().rss
            except (psutil.NoSuchProcess,psutil.AccessDenied): pass
        return vm.percent,vm.available,rss
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Install memory monitoring dependency: python -m pip install psutil')
    mem={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        k,v=line.split(':',1);mem[k]=int(v.split()[0])*1024
    total=mem['MemTotal'];available=mem['MemAvailable'];ppids={};sizes={}
    for d in Path('/proc').iterdir():
        if not d.name.isdigit(): continue
        try:
            stat=(d/'stat').read_text().rsplit(')',1)[1].split()
            ppids[int(d.name)]=int(stat[1])
            sizes[int(d.name)]=int((d/'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
        except (OSError,ValueError,IndexError): continue
    own={os.getpid()}
    while True:
        more={pid for pid,ppid in ppids.items() if ppid in own}-own
        if not more: break
        own|=more
    return 100*(total-available)/total,available,sum(sizes.get(pid,0) for pid in own)
