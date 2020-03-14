import NvRules

def get_identifier():
    return "CPIStallLgThrottle"

def get_name():
    return "CPI Stall 'LG Throttle'"

def get_description():
    return "Warp stall analysis for 'LG Throttle' issues"

def get_section_identifier():
    return "WarpStateStats"

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    isSupported = False
    ccMajor = action.metric_by_name("device__attribute_compute_capability_major").as_uint64()
    ccMinor = action.metric_by_name("device__attribute_compute_capability_minor").as_uint64()
    if ccMajor == 7 and ccMinor >= 0:
        isSupported = True
        issueActive = action.metric_by_name("smsp__issue_active.avg.per_cycle_active").as_double()
        warpCyclesPerStall = action.metric_by_name("smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio").as_double()
        warpCyclesPerIssue = action.metric_by_name("smsp__average_warps_active_per_issue_active.ratio").as_double()

    if isSupported and issueActive < 0.8 and 0.3 < (warpCyclesPerStall / warpCyclesPerIssue):
        message = "On average each warp of this kernel spends {:.1f} cycles being stalled waiting for the local/global instruction queue to be not full. This represents about {:.1f}% of the total average of {:.1f} cycles between issuing two instructions. Typically this stall occurs only when executing local or global memory instructions extremely frequently. If applicable, consider combining multiple lower-width memory operations into fewer wider memory operations and try interleaving memory operations and math instructions.".format(warpCyclesPerStall, 100.*warpCyclesPerStall/warpCyclesPerIssue, warpCyclesPerIssue)

        fe.message(NvRules.IFrontend.MsgType_MSG_WARNING, message)

