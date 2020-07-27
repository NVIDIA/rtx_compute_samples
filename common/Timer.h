/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <chrono>

/**
 * Simple Timer class
 */
class Timer {
public:
  using timer_t = std::chrono::high_resolution_clock;
  using time_point_t = timer_t::time_point;

public:
  Timer() { restart(); }

  void restart(void) {
    m_start = timer_t::now();
    m_running = true;
  }

  void stop(void) {
    if (m_running)
      m_stop = timer_t::now();
  }

  double get_elapsed_s(void) {
    return get_elapsed<std::chrono::nanoseconds>().count() * 1e-9;
  }

  double get_elapsed_ms(void) {
    return get_elapsed<std::chrono::nanoseconds>().count() * 1e-6;
  }

  double get_elapsed_us(void) {
    return get_elapsed<std::chrono::nanoseconds>().count() * 1e-3;
  }

  double get_elapsed_ns(void) {
    return get_elapsed<std::chrono::nanoseconds>().count() * 1e-0;
  }

private:
  bool m_running = false;
  time_point_t m_start, m_stop;

  template <typename Duration> Duration get_elapsed(void) {
    time_point_t stop = (m_running) ? timer_t::now() : m_stop;
    return std::chrono::duration_cast<Duration>(stop - m_start);
  }
};
