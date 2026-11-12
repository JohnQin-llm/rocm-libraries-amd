// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tests for the per-solution time limit of the Find mini-benchmark.
//
// EvaluateInvokers (solver_finders.cpp) times each candidate solution with up to 8 runs
// (one warm-up), and stops early once about 5 s of kernel time has been spent on it. Without
// that limit a slow candidate is launched 8 times: on gfx950, ConvDirectNaiveConvFwd takes
// ~20 s per launch for a channels-last (NDHWC) 3D convolution such as
// N=1 C=256 8x128x128 K=512 3x4x4 stride 2 (fp16/bf16/fp32), so Find spent ~170 s on a
// solution that was never going to be selected, and looked like a hang.
//
// The invokers used here launch no kernels: like a profiled kernel launch, they only store
// their run time in the handle, which is what EvaluateInvokers reads.

#include <gtest/gtest.h>

#include <miopen/conv/solver_finders.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/handle.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/names.hpp>
#include <miopen/solution.hpp>

#include "get_handle.hpp"

#include <string>
#include <vector>

namespace {

struct FakeTiming
{
    float run_ms;
    int runs = 0;
};

miopen::solver::ConvSolution MakeTimedSolution(const std::string& solver_id, FakeTiming& timing)
{
    auto sol            = miopen::solver::ConvSolution{miopenStatusSuccess};
    sol.solver_id       = solver_id;
    sol.workspace_sz    = 0;
    sol.invoker_factory = [&timing](const std::vector<miopen::Kernel>&) {
        return [&timing](const miopen::Handle& handle, const miopen::AnyInvokeParams&) {
            ++timing.runs;
            handle.ResetKernelTime();
            handle.AccumKernelTime(timing.run_ms);
        };
    };
    return sol;
}

std::vector<miopen::Solution> Evaluate(const miopen::solver::ConvSolution& sol)
{
    auto&& handle            = get_handle();
    auto core_result         = miopen::FindCoreResult{};
    core_result.is_optimal   = true;
    bool non_naive_succeeded = false;
    return miopen::EvaluateInvokers(handle,
                                    {sol},
                                    miopen::AlgorithmName{"miopenConvolutionFwdAlgoGEMM"},
                                    miopen::NetworkConfig{"conv_find_time_limit_test"},
                                    miopen::AnyInvokeParams{},
                                    core_result,
                                    false,
                                    non_naive_succeeded);
}

} // namespace

// A fast solution gets the full 8 runs (1 warm-up + 7 samples).
TEST(GPU_ConvFindTimeLimit_FP32, FastSolutionRunsEightTimes)
{
    FakeTiming timing{0.5f};
    const auto results = Evaluate(MakeTimedSolution("GemmFwdRest", timing));
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(timing.runs, 8);
    EXPECT_FLOAT_EQ(results[0].GetTime(), 0.5f);
}

// 2 s per run: warm-up + 2 samples reach the 5 s limit, so only 3 runs are made.
TEST(GPU_ConvFindTimeLimit_FP32, SlowSolutionStopsAtTimeLimit)
{
    FakeTiming timing{2000.0f};
    const auto results = Evaluate(MakeTimedSolution("GemmFwdRest", timing));
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(timing.runs, 3);
    EXPECT_FLOAT_EQ(results[0].GetTime(), 2000.0f);
}

// A single run already exceeds the limit: only the warm-up run is made, and its time is used.
TEST(GPU_ConvFindTimeLimit_FP32, VerySlowSolutionRunsOnce)
{
    FakeTiming timing{20000.0f};
    const auto results = Evaluate(MakeTimedSolution("ConvDirectNaiveConvFwd", timing));
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(timing.runs, 1);
    EXPECT_FLOAT_EQ(results[0].GetTime(), 20000.0f);
}
