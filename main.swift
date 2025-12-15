//
//  main.swift
//  quantumsort
//
//  created by Harri Hilding Smatt on 2025-12-24
//

import Foundation
import Metal

print("Hello, World!")

var array_index: UInt32 = 0
var array_size: UInt32 = 1000000
var to_sort: Array<UInt32> = Array(repeating: 0, count: Int(array_size))

while array_index < array_size {
    to_sort[Int(array_index)] = UInt32.random(in: 0..<0xFFFFFFFF)
    array_index += 1
}

let mtl_device: MTLDevice = MTLCreateSystemDefaultDevice()!

if mtl_device.supportsFamily(MTLGPUFamily.metal4) {
    let mtl_command_queue = mtl_device.makeCommandQueue()!
    let mtl_command_buffer = mtl_command_queue.makeCommandBuffer()!
    let mtl_library = mtl_device.makeDefaultLibrary()!
    let mtl_function_sort = mtl_library.makeFunction(name: "quantumsort")!
    let mtl_buffer_to_sort: MTLBuffer = mtl_device.makeBuffer(bytes: to_sort, length: MemoryLayout<Int32>.size * Int(array_size))!
    let mtl_buffer_sorted: MTLBuffer = mtl_device.makeBuffer(bytes: to_sort, length: MemoryLayout<Int32>.size * Int(array_size))!
    let mtl_buffer_sorted_temp: MTLBuffer = mtl_device.makeBuffer(bytes: to_sort, length: MemoryLayout<Int32>.size * Int(array_size))!

    do {
        let mtl_pipeline_state = try mtl_device.makeComputePipelineState(function: mtl_function_sort)
        let mtl_command_encoder = mtl_command_buffer.makeComputeCommandEncoder()!

        mtl_command_encoder.setComputePipelineState(mtl_pipeline_state)
        mtl_command_encoder.setBuffer(mtl_buffer_to_sort, offset: 0, index: 0)
        mtl_command_encoder.setBuffer(mtl_buffer_sorted, offset: 0, index: 1)
        mtl_command_encoder.setBuffer(mtl_buffer_sorted_temp, offset: 0, index: 2)
        mtl_command_encoder.setBytes(&array_size, length: MemoryLayout<UInt32>.size, index: 3)
        mtl_command_encoder.dispatchThreads(MTLSizeMake(mtl_pipeline_state.threadExecutionWidth, 1, 1), threadsPerThreadgroup: MTLSizeMake(mtl_pipeline_state.threadExecutionWidth, 1, 1))
        mtl_command_encoder.endEncoding()

        print("Simd width: \(mtl_pipeline_state.threadExecutionWidth)")
    } catch let error {
        print(error.localizedDescription)
    }
    
    mtl_command_buffer.commit()
    mtl_command_buffer.waitUntilCompleted()
    
    print("Time to sort: \(mtl_command_buffer.kernelEndTime - mtl_command_buffer.kernelStartTime)s")
    
    let sorted_pointer = mtl_buffer_sorted.contents()
    let sorted = sorted_pointer.bindMemory(to: UInt32.self, capacity: Int(array_size))

    for index in 1..<Int(array_size) {
        if sorted[index - 1] > sorted[index] {
            fatalError("Sorting failed")
        }
    }
}
else {
    print("Metal 4 not available")
}

print("End.")
exit(42)
