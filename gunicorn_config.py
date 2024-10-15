# From vits-simple-api

import gc
import multiprocessing

bind = "0.0.0.0:55557"
# workers = multiprocessing.cpu_count()
workers = 1
# preload_app = True # This breaks breaks everything, don't comment this out
timeout = 300 # Long timeout to download larger models

# disable GC in master as early as possible
gc.disable()

def when_ready(server):
    # freeze objects after preloading app
    gc.freeze()
    print("Objects frozen in perm gen: ", gc.get_freeze_count())

def post_fork(server, worker):
    # reenable GC on worker
    gc.enable()