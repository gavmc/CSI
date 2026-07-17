#pragma once
#include <cstdint>

class Pinger {
public:
    bool start(const char* target_ip, uint32_t interval_ms);
};