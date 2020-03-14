import NvRules

def get_identifier():
    return "SlowPipeLimiter"

def get_name():
    return "Slow Pipe Limiter"

def get_description():
    return "Slow pipe limiting compute utilization"

def get_section_identifier():
    return "ComputeWorkloadAnalysis"

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    cc_major = action.metric_by_name("device__attribute_compute_capability_major").as_uint64()
    cc_minor = action.metric_by_name("device__attribute_compute_capability_minor").as_uint64()
    if cc_major == 7 and cc_minor >= 0:
        sm_busy = action.metric_by_name("sm__instruction_throughput.avg.pct_of_peak_sustained_active").as_double()
        inst_issued_avg = action.metric_by_name("sm__inst_issued.avg.pct_of_peak_sustained_active").as_double()
        inst_issued_max = action.metric_by_name("sm__inst_issued.max.pct_of_peak_sustained_active").as_double()

        no_bound_threshold = 80
        issued_avg_threshold = 20
        diff_threshold = 25

        pipe_diff = inst_issued_max - inst_issued_avg
        if sm_busy >= no_bound_threshold and inst_issued_avg < issued_avg_threshold and pipe_diff > diff_threshold:
            fe.message("It is possible that a slow pipeline is preventing better kernel performance. The average pipeline utilization of {:.1f}% is {:.1f}% lower than the maximum utilization of {:1.f}%. Try moving compute to other pipelines, e.g. from fp64 to fp32 or int.".format(inst_issued_avg, pipe_diff, inst_issued_max))

