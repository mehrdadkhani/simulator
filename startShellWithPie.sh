#mm-link trace10 trace10 --uplink-queue=droptail --downlink-queue=infinite --uplink-queue-args="packets=10000"
mm-link trace10 trace100 --uplink-queue=pie --downlink-queue=infinite --uplink-queue-args="packets=300, qdelay_ref=50, max_burst=1" mm-delay 10

