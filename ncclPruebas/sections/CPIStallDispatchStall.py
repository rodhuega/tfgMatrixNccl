import NvRules

def get_identifier():
    return "CPIStallDispatchStall"

def get_name():
    return "CPI Stall 'Dispatch Stall'"

def get_description():
    return "Warp stall analysis for 'Dispatch Stall' issues"

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
        warpCyclesPerStall = action.metric_by_name("smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio").as_double()
        warpCyclesPerIssue = action.metric_by_name("smsp__average_warps_active_per_issue_active.ratio").as_double()
    else:
        issueActive = action.metric_by_name("smsp__issue_active_avg_per_active_cycle").as_double()
        warpCyclesPerStall = action.metric_by_name("smsp__warp_cycles_per_issue_stall_dispatch_stall").as_double()
        warpCyclesPerIssue = action.metric_by_name("smsp__warp_cycles_per_issue_active").as_double()

    if issueActive < 0.8 and 0.3 < (warpCyclesPerStall / warpCyclesPerIssue):
        message = "On average each warp of this kernel spends {:.1f} cycles being stalled waiting on a dispatch stall. This represents about {:.1f}% of the total average of {:.1f} cycles between issuing two instructions. A warp stalled during dispatch has a ready instruction, but the dispatcher holds back issuing the warp due to other conflicts or events.".format(warpCyclesPerStall, 100.*warpCyclesPerStall/warpCyclesPerIssue, warpCyclesPerIssue)

        fe.message(NvRules.IFrontend.MsgType_MSG_WARNING, message)

