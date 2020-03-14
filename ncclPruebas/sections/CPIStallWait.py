import NvRules

def get_identifier():
    return "CPIStallWait"

def get_name():
    return "CPI Stall 'Wait'"

def get_description():
    return "Warp stall analysis for 'Wait' issues"

def get_section_identifier():
    return "WarpStateStats"

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    ccMajor = action.metric_by_name("device__attribute_compute_capability_major").as_uint64()
    ccMinor = action.metric_by_name("device__attribute_compute_capability_minor").as_uint64()
    if ccMajor == 7 and ccMinor >= 0:
        issueActive = action.metric_by_name("smsp__issue_active.avg.per_cycle_active").as_double()
        warpCyclesPerStall = action.metric_by_name("smsp__average_warps_issue_stalled_wait_per_issue_active.ratio").as_double()
        warpCyclesPerIssue = action.metric_by_name("smsp__average_warps_active_per_issue_active.ratio").as_double()
    else:
        issueActive = action.metric_by_name("smsp__issue_active_avg_per_active_cycle").as_double()
        warpCyclesPerStall = action.metric_by_name("smsp__warp_cycles_per_issue_stall_wait").as_double()
        warpCyclesPerIssue = action.metric_by_name("smsp__warp_cycles_per_issue_active").as_double()

    if issueActive < 0.8 and 0.3 < (warpCyclesPerStall / warpCyclesPerIssue):
        message = "On average each warp of this kernel spends {:.1f} cycles being stalled on a fixed latency execution dependency. This represents about {:.1f}% of the total average of {:.1f} cycles between issuing two instructions. Typically, this stall reason should be very low and only shows up as a top contributor in already highly optimized kernels. If possible, try to further increase the number of active warps to hide the corresponding instruction latencies.".format(warpCyclesPerStall, 100.*warpCyclesPerStall/warpCyclesPerIssue, warpCyclesPerIssue)

        fe.message(NvRules.IFrontend.MsgType_MSG_WARNING, message)

