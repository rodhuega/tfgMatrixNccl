import NvRules

def get_identifier():
    return "CPIStallNoInstructions"

def get_name():
    return "CPI Stall 'No Instructions'"

def get_description():
    return "Warp stall analysis for 'No Instructions' issues"

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
        warpCyclesPerStall = action.metric_by_name("smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio").as_double()
        warpCyclesPerIssue = action.metric_by_name("smsp__average_warps_active_per_issue_active.ratio").as_double()
    else:
        issueActive = action.metric_by_name("smsp__issue_active_avg_per_active_cycle").as_double()
        warpCyclesPerStall = action.metric_by_name("smsp__warp_cycles_per_issue_stall_no_instructions").as_double()
        warpCyclesPerIssue = action.metric_by_name("smsp__warp_cycles_per_issue_active").as_double()

    if issueActive < 0.8 and 0.3 < (warpCyclesPerStall / warpCyclesPerIssue):
        message = "On average each warp of this kernel spends {:.1f} cycles being stalled due to not having the next instruction fetched yet. This represents about {:.1f}% of the total average of {:.1f} cycles between issuing two instructions. A high number of warps not having an instruction fetched is typical for very short kernels with less than one full wave of work in the grid. Excessively jumping across large blocks of assembly code can also lead to more warps stalled for this reason.".format(warpCyclesPerStall, 100.0*warpCyclesPerStall/warpCyclesPerIssue, warpCyclesPerIssue)

        fe.message(NvRules.IFrontend.MsgType_MSG_WARNING, message)

