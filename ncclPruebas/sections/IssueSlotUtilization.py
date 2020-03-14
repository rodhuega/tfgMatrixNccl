import NvRules

def get_identifier():
    return "IssueSlotUtilization"

def get_name():
    return "Issue Slot Utilization"

def get_description():
    return "Scheduler instruction issue analysis"

def get_section_identifier():
    return "SchedulerStats"

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    ccMajor = action.metric_by_name("device__attribute_compute_capability_major").as_uint64()
    ccMinor = action.metric_by_name("device__attribute_compute_capability_minor").as_uint64()
    if ccMajor == 7 and ccMinor >= 0:
        issueActive = action.metric_by_name("smsp__issue_active.avg.per_cycle_active").as_double()
        activeWarps = action.metric_by_name("smsp__warps_active.avg.per_cycle_active").as_double()
        eligibleWarps = action.metric_by_name("smsp__warps_eligible.avg.per_cycle_active").as_double()
        maxWarps = action.metric_by_name("smsp__warps_active.avg.peak_sustained").as_double()
    else:
        issueActive = action.metric_by_name("smsp__issue_active_avg_per_active_cycle").as_double()
        activeWarps = action.metric_by_name("smsp__active_warps_avg_per_active_cycle").as_double()
        eligibleWarps = action.metric_by_name("smsp__eligible_warps_avg_per_active_cycle").as_double()
        maxWarps = action.metric_by_name("smsp__warps_per_cycle_max").as_double()

    if ccMajor == 7:
        instPerCycle = "one instruction"
        issueActiveTarget = 0.6
    else:
        instPerCycle = "two instructions"
        issueActiveTarget = 0.8

    if issueActive < issueActiveTarget:
        message = "Every scheduler is capable of issuing {} per cycle, but for this kernel each scheduler only issues an instruction every {:.1f} cycles. This might leave hardware resources underutilized and may lead to less optimal performance.".format(instPerCycle, 1./issueActive)
        message += " Out of the maximum of {} warps per scheduler, this kernel allocates an average of {:.2f} active warps per scheduler,".format(int(maxWarps), activeWarps)

        if activeWarps < 1.0:
            message += " which already limits the scheduler to less than a warp per instruction. Try to increase the number of active warps by increasing occupancy and/or avoid possible load imbalances due to highly different execution durations per warp."
        else:
            message += " but only an average of {:.2f} warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible warp results in no instruction being issued and the issue slot remains unused. To increase the number of eligible warps either increase the number of active warps or reduce the time the active warps are stalled.".format(eligibleWarps)

        fe.message(NvRules.IFrontend.MsgType_MSG_WARNING, message)

