import NvRules
import math

def get_identifier():
    return "LaunchConfiguration"

def get_name():
    return "Launch Configuration"

def get_description():
    return "Kernel launch configuration analysis"

def get_section_identifier():
    return "LaunchStats"

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    block_size = action.metric_by_name("launch__block_size").as_double()
    grid_size = action.metric_by_name("launch__grid_size").as_double()
    num_sms = action.metric_by_name("device__attribute_multiprocessor_count").as_uint64()
    num_waves = action.metric_by_name("launch__waves_per_multiprocessor").as_double()

    ccMajor = action.metric_by_name("device__attribute_compute_capability_major").as_uint64()
    ccMinor = action.metric_by_name("device__attribute_compute_capability_minor").as_uint64()
    if ccMajor == 7 and ccMinor >= 0:
        achieved_occ = action.metric_by_name("sm__warps_active.avg.pct_of_peak_sustained_active").as_double()
    else:
        achieved_occ = action.metric_by_name("sm__active_warps_avg_per_active_cycle_pct").as_double()

    theoretical_occ = action.metric_by_name("sm__maximum_warps_per_active_cycle_pct").as_double()

    if block_size % 32 != 0:
        fe.message(NvRules.IFrontend.MsgType_MSG_WARNING, "Threads are executed in groups of 32 threads called warps. This kernel launch is configured to execute {:d} threads per block. Consequently, some threads in a warp are masked off and those hardware resources are unused. Try changing the number of threads per block to be a multiple of 32 threads. Between 128 and 256 threads per block is a good initial range for experimentation. Use smaller thread blocks rather than one large thread block per multiprocessor if latency affects performance. This is particularly beneficial to kernels that frequently call __syncthreads().".format(int(block_size)))

    if grid_size < num_sms:
        fe.message(NvRules.IFrontend.MsgType_MSG_WARNING, "The grid for this launch is configured to execute only {:d} blocks, which is less than the GPU's {:d} multiprocessors. This can underutilize some multiprocessors. If you do not intend to execute this kernel concurrently with other workloads, consider reducing the block size to have at least one block per multiprocessor or increase the size of the grid to fully utilize the available hardware resources.".format(int(grid_size), int(num_sms)))
    elif grid_size < 2 * num_sms:
        fe.message(NvRules.IFrontend.MsgType_MSG_WARNING, "If you execute __syncthreads() to synchronize the threads of a block, it is recommended to have more than the achieved {:d} blocks per multiprocessor. This way, blocks that aren't waiting for __syncthreads() can keep the hardware busy.".format(int(grid_size / num_sms)))

    partial_waves, whole_waves = math.modf(num_waves)
    partial_wave_blocks = int(grid_size * (partial_waves / num_waves))
    potential_tail_effect = 0. if partial_waves == 0 else 1. / (whole_waves + 1.)
    if whole_waves >= 1. and potential_tail_effect >= 0.2 and achieved_occ < theoretical_occ * 0.8:
        fe.message(NvRules.IFrontend.MsgType_MSG_WARNING, "A wave of thread blocks is defined as the maximum number of blocks that can be executed in parallel on the target GPU. The number of blocks in a wave depends on the number of multiprocessors and the theoretical occupancy of the kernel. This kernel launch results in {:d} full waves and a partial wave of {:d} thread blocks. Under the assumption of a uniform execution duration of all thread blocks, the partial wave may account for up to {:.1f}% of the total kernel runtime with a lower occupancy of {:.1f}%. Try launching a grid with no partial wave. The overall impact of this tail effect also lessens with the number of full waves executed for a grid.".format(int(whole_waves), partial_wave_blocks, 100. * potential_tail_effect, 100. * (theoretical_occ - achieved_occ) / theoretical_occ))
