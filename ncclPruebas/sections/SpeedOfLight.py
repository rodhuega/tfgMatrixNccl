import NvRules
import math

def get_identifier():
    return "SOLBottleneck"

def get_name():
    return "Bottleneck"

def get_description():
    return "High-level bottleneck detection"

def get_section_identifier():
    return "SpeedOfLight"

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    ccMajor = action.metric_by_name("device__attribute_compute_capability_major").as_uint64()
    ccMinor = action.metric_by_name("device__attribute_compute_capability_minor").as_uint64()
    num_waves = action.metric_by_name("launch__waves_per_multiprocessor").as_double()
    if ccMajor == 7 and ccMinor >= 0:
        smSolPct = action.metric_by_name("sm__throughput.avg.pct_of_peak_sustained_elapsed").as_double()
        memSolPct = action.metric_by_name("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed").as_double()
    else:
        smSolPct = action.metric_by_name("sm__sol_pct").as_double()
        memSolPct = action.metric_by_name("gpu__compute_memory_sol_pct").as_double()

    balanced_threshold = 10
    latency_bound_threshold = 60
    no_bound_threshold = 80
    waves_threshold = 1

    msg_type = NvRules.IFrontend.MsgType_MSG_OK

    if smSolPct < no_bound_threshold and memSolPct < no_bound_threshold:
        if smSolPct < latency_bound_threshold and memSolPct < latency_bound_threshold:
            msg_type = NvRules.IFrontend.MsgType_MSG_WARNING
            if num_waves < waves_threshold:
                message = "This kernel grid is too small to fill the available resources on this device. Look at `Launch Statistics` for more details."
            else:
                message = "This kernel exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below {:.1f}% of peak typically indicate latency issues. Look at `Scheduler Statistics` and `Warp State Statistics` for potential reasons.".format(latency_bound_threshold)
        elif math.fabs(smSolPct - memSolPct) >= balanced_threshold:
            msg_type = NvRules.IFrontend.MsgType_MSG_WARNING
            if smSolPct > memSolPct:
                message = "Compute is more heavily utilized than Memory: Look at `Compute Workload Analysis` report section to see what the compute pipelines are spending their time doing. Also, consider whether any computation is redundant and could be reduced or moved to look-up tables."
            else:
                message = "Memory is more heavily utilized than Compute: Look at `Memory Workload Analysis` report section to see where the memory system bottleneck is. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the bytes transferred. Also consider whether it is possible to do more work per memory access (kernel fusion) or whether there are values you can (re)compute."
        else:
            message = "Compute and Memory are well-balanced: To reduce runtime, both computation and memory traffic must be reduced. Check both the `Compute Workload Analysis` and `Memory Workload Analysis` report sections."
    else:
        message = "The kernel is utilizing greater than {:.1f}% of the available compute or memory performance of the device. To further improve performance, work will likely need to be shifted from the most utilized to another unit.".format(no_bound_threshold)

    fe.message(msg_type, message)
