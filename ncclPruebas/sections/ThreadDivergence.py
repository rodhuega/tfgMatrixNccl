import NvRules

def get_identifier():
    return "ThreadDivergence"

def get_name():
    return "Thread Divergence"

def get_description():
    return "Warp and thread control flow analysis"

def get_section_identifier():
    return "WarpStateStats"

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    ccMajor = action.metric_by_name("device__attribute_compute_capability_major").as_uint64()
    ccMinor = action.metric_by_name("device__attribute_compute_capability_minor").as_uint64()
    if ccMajor == 7 and ccMinor >= 0:
        threadInstExecuted = action.metric_by_name("smsp__thread_inst_executed_per_inst_executed.ratio").as_double()
        threadInstExecutedNPO = action.metric_by_name("smsp__thread_inst_executed_pred_on_per_inst_executed.ratio").as_double()
    else:
        threadInstExecuted = action.metric_by_name("smsp__thread_inst_executed_per_inst_executed").as_double()
        threadInstExecutedNPO = action.metric_by_name("smsp__thread_inst_executed_not_pred_off_per_inst_executed").as_double()

    if threadInstExecuted < 24. or threadInstExecutedNPO < 24:
        message = "Instructions are executed in warps, which are groups of 32 threads. Optimal instruction throughput is achieved if all 32 threads of a warp execute the same instruction. The chosen launch configuration, early thread completion, and divergent flow control can signifcantly lower the number of active threads in a warp per cycle. This kernel achieves an average of {0:.1f} threads being active per cycle.".format(threadInstExecuted)

        if threadInstExecutedNPO < threadInstExecuted:
            message += " This is further reduced to {0:.1f} threads per warp due to predication. The compiler may use predication to avoid an actual branch. Instead, all instructions are scheduled, but a per-thread condition code or predicate controls which threads execute the instructions. Try to avoid different execution paths within a warp when possible.".format(threadInstExecutedNPO)

            if ccMajor == 7:
                message += " In addition, assure your kernel makes use of Independent Thread Scheduling, which allows a warp to reconverge after a data-dependent conditional block by explicitely calling __syncwarp()."

        fe.message(NvRules.IFrontend.MsgType_MSG_WARNING, message)
