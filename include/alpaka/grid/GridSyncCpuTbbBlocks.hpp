/* Copyright 2024 Mykhailo Varvarin
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/grid/Traits.hpp"

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
#    include "alpaka/core/BarrierTbb.h"

namespace alpaka
{
    //! The thread id map barrier grid synchronization for TBB.
    template<typename TIdx>
    class GridSyncBarrierTbb : public interface::Implements<ConceptGridSync, GridSyncBarrierTbb<TIdx>>
    {
    public:
        using Barrier = core::tbb::BarrierThread<TIdx>;

        ALPAKA_FN_HOST explicit GridSyncBarrierTbb(TIdx const& gridThreadCount) : m_barrier(gridThreadCount)
        {
        }

        Barrier mutable m_barrier;
    };

    namespace trait
    {
        template<typename TIdx>
        struct SyncGridThreads<GridSyncBarrierTbb<TIdx>>
        {
            ALPAKA_FN_HOST static auto syncGridThreads(GridSyncBarrierTbb<TIdx> const& gridSync) -> void
            {
                gridSync.m_barrier.wait();
            }
        };

    } // namespace trait
} // namespace alpaka

#endif
