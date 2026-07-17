#pragma once
#include <cstdint>
#include "config.hpp"

constexpr uint16_t CSI_RECORD_MAGIC = 0xC517;

struct __attribute__((packed)) CsiRecord{
    uint16_t magic;             // CSI_RECORD_MAGIC
    uint8_t  sensor_id;
    uint8_t  reserved0;
    uint32_t seq;
    int64_t  timestamp_us;      // esp_timer_get_time() at capture
    int8_t   rssi;
    int8_t   noise_floor;
    uint8_t  channel;
    uint8_t  secondary_channel;
    uint8_t  src_mac[6];
    uint8_t  reserved1[2];
    uint16_t buf_len;
    uint8_t  first_word_invalid;
    uint8_t  reserved2;
    int8_t   buf[CSI_BUF_MAX];
};