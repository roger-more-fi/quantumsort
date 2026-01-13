//
//  sort.metal
//  quantumsort
//
//  created by Harri Hilding Smatt on 2025-12-24
//

#include <metal_stdlib>
using namespace metal;

void quantumsort_iter(const device uint32_t* keys_to_sort,
                      device uint32_t* keys_sorted_in,
                      device uint32_t* keys_sorted_out,
                      const uint32_t keys_sz,
                      const uint32_t thread_id,
                      const uint32_t warp_sz,
                      const int radix)
{
    // prefixsum
    uint32_t prefixsum[16] = { 0 };
    // calculate each 16bit counts into separate threads
    for (uint32_t index = thread_id; index < keys_sz; index += warp_sz)
        ++prefixsum[(keys_to_sort[index] >> radix) & 0x0F];
    // share bit counts from each thread for same sum in each thread
    for (uint32_t index = 0; index < 16; ++index)
        prefixsum[index] = simd_sum(prefixsum[index]);
    // calculate one prefix sum in each thread
    for (uint32_t index = thread_id; index < 16; index += warp_sz)
        prefixsum[index] = simd_prefix_exclusive_sum(prefixsum[index]);
    // broacast prefix sums for each thread
    for (uint32_t index = 0; index < 16; ++index)
        prefixsum[index] = simd_broadcast(prefixsum[index], index);
    // shuffle
    for (uint32_t index = thread_id; index < keys_sz; index += warp_sz) {
        const uint32_t key_index = keys_sorted_in[index];
        const uint32_t key_value = keys_to_sort[key_index];
        const uint32_t key_masked = (key_value >> radix) & 0x0F;
        uint32_t key_sorted_index = 0xDFDF;
        for (uint32_t prefixsum_index = 0; prefixsum_index < 16; ++prefixsum_index) {
            const uint32_t is_match = key_masked == prefixsum_index;
            const uint32_t match_count = simd_sum(is_match);
            if (is_match) {
                key_sorted_index = simd_prefix_exclusive_sum(is_match);
                key_sorted_index += prefixsum[prefixsum_index];
            }
            prefixsum[prefixsum_index] += match_count;
        }
        keys_sorted_out[key_sorted_index] = key_index;
    }
}

kernel void quantumsort(const device uint32_t* keys_to_sort [[ buffer(0) ]],
                        device uint32_t* keys_sorted [[ buffer(1) ]],
                        device uint32_t* keys_sorted_temp [[ buffer(2) ]],
                        const device uint32_t& keys_sz [[ buffer(3) ]],
                        const uint32_t thread_id [[ thread_index_in_simdgroup ]],
                        const uint32_t warp_sz [[ threads_per_simdgroup ]])
{
    // init
    for (uint32_t index = thread_id; index < keys_sz; index += warp_sz)
        keys_sorted[index] = index;
    // main loop
    for (int radix = 0; radix < 32; radix += 8) {
        quantumsort_iter(keys_to_sort, keys_sorted, keys_sorted_temp, keys_sz, thread_id, warp_sz, radix + 0);
        quantumsort_iter(keys_to_sort, keys_sorted_temp, keys_sorted, keys_sz, thread_id, warp_sz, radix + 4);
    }
    // shuffle
    for (uint32_t index = thread_id; index < keys_sz; index += warp_sz)
        keys_sorted[index] = keys_to_sort[keys_sorted[index]];
}
