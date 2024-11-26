/* Copyright 2024 Mykhailo Varvarin
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <cstdint>
#include <iostream>

//! Hello world kernel, utilizing grid synchronization.
//! Prints hello world from a thread, performs grid sync.
//! and prints the sum of indixes of this thread and the opposite thread (the sums have to be the same).
//! Prints an error if sum is incorrect.
struct HelloWorldKernel
{
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const& acc, size_t* array, bool* success) const
    {
        // Get index of the current thread in the grid and the total number of threads.
        size_t gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        size_t gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];

        if(gridThreadIdx == 0)
            printf("Hello, World from alpaka thread %zu!\n", gridThreadIdx);

        // Write the index of the thread to array.
        array[gridThreadIdx] = gridThreadIdx;

        // Perform grid synchronization.
        alpaka::syncGridThreads(acc);

        // Get the index of the thread from the opposite side of 1D array.
        size_t gridThreadIdxOpposite = array[gridThreadExtent - gridThreadIdx - 1];

        // Sum them.
        size_t sum = gridThreadIdx + gridThreadIdxOpposite;

        // Get the expected sum.
        size_t expectedSum = gridThreadExtent - 1;

        // Print the result and signify an error if the grid synchronization fails.
        if(sum != expectedSum)
        {
            *success = false;
            printf(
                "After grid sync, this thread is %zu, thread on the opposite side is %zu. Their sum is %zu, expected: "
                "%zu.%s",
                gridThreadIdx,
                gridThreadIdxOpposite,
                sum,
                expectedSum,
                sum == expectedSum ? "\n" : " ERROR: the sum is incorrect.\n");
        }
    }
};

// In standard projects, you typically do not execute the code with any available accelerator.
// Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
// selected accelerator only. If you use the example as the starting point for your project, you can rename the
// example() function to main() and move the accelerator tag to the function body.
template<typename TAccTag>
auto example(TAccTag const&) -> int
{
    // Define the accelerator
    // For simplicity this examples always uses 1 dimensional indexing, and index type size_t
    using Acc = alpaka::TagToAcc<TAccTag, alpaka::DimInt<1>, std::size_t>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    // Define dimensionality and type of indices to be used in kernels
    using Dim = alpaka::DimInt<1>;
    using Idx = size_t;


    // Select the first device available on a system, for the chosen accelerator
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = getDevByIdx(platformAcc, 0u);

    // Select CPU host
    constexpr auto platformHost = alpaka::Platform<alpaka::DevCpu>{};
    auto const devHost = getDevByIdx(platformHost, 0u);

    // Define type for a queue with requested properties: Blocking.
    using Queue = alpaka::Queue<Acc, alpaka::Blocking>;
    // Create a queue for the device.
    auto queue = Queue{devAcc};

    // Define kernel execution configuration of blocks,
    // threads per block, and elements per thread.
    Idx blocksPerGrid = 100;
    Idx threadsPerBlock = 1;
    Idx elementsPerThread = 1;

    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;

    // Allocate memory on the device.
    alpaka::Vec<Dim, Idx> bufferExtent{blocksPerGrid * threadsPerBlock};
    auto deviceMemory = alpaka::allocBuf<Idx, Idx>(devAcc, bufferExtent);

    // Allocate the result value
    auto bufAccResult = alpaka::allocBuf<bool, Idx>(devAcc, static_cast<Idx>(1u));
    memset(queue, bufAccResult, static_cast<std::uint8_t>(true));


    // Instantiate the kernel object.
    HelloWorldKernel helloWorldKernel;

    // Query the maximum number of blocks allowed for the device
    int maxBlocks = alpaka::getMaxActiveBlocks<Acc>(
        devAcc,
        helloWorldKernel,
        threadsPerBlock,
        elementsPerThread,
        getPtrNative(deviceMemory),
        getPtrNative(bufAccResult));
    std::cout << "Maximum blocks for the kernel: " << maxBlocks << std::endl;

    // Create a workdiv according to the limitations
    blocksPerGrid = std::min(static_cast<Idx>(maxBlocks), blocksPerGrid);
    auto workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};

    // Create a task to run the kernel.
    // Note the cooperative kernel specification.
    // Only cooperative kernels can perform grid synchronization.
    auto taskRunKernel = alpaka::createTaskCooperativeKernel<Acc>(
        workDiv,
        helloWorldKernel,
        getPtrNative(deviceMemory),
        getPtrNative(bufAccResult));

    // Enqueue the kernel execution task.
    alpaka::enqueue(queue, taskRunKernel);

    // Copy the result value to the host
    auto bufHostResult = alpaka::allocBuf<bool, Idx>(devHost, static_cast<Idx>(1u));
    memcpy(queue, bufHostResult, bufAccResult);
    wait(queue);

    auto const result = *getPtrNative(bufHostResult);

    if(result)
    {
        return EXIT_SUCCESS;
    }
    else
    {
        return EXIT_FAILURE;
    }
}

auto main() -> int
{
    // Execute the example once for each enabled accelerator.
    // If you would like to execute it for a single accelerator only you can use the following code.
    //  \code{.cpp}
    //  auto tag = TagCpuSerial;
    //  return example(tag);
    //  \endcode
    //
    // valid tags:
    //   TagCpuSerial, TagGpuHipRt, TagGpuCudaRt, TagCpuOmp2Blocks, TagCpuTbbBlocks,
    //   TagCpuOmp2Threads, TagCpuSycl, TagCpuTbbBlocks, TagCpuThreads,
    //   TagFpgaSyclIntel, TagGenericSycl, TagGpuSyclIntel
    return alpaka::executeForEachAccTag([=](auto const& tag) { return example(tag); });
}
