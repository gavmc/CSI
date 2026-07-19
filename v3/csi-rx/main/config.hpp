#pragma once
#include <cstdint>

constexpr const char* WIFI_SSID     = "NETGEAR82";
constexpr const char* WIFI_PASSWORD = "slowcanoe599";

constexpr const char* COLLECTOR_IP = "192.168.1.5";
constexpr uint16_t CSI_PORT  = 5566;   // CSI data -> collector
constexpr uint16_t PING_PORT = 4444;   // ping/sync -> echo server on desktop

constexpr uint8_t SENSOR_ID = 4;       // 1, 2, 3, or 4 (MUST be unique!)

constexpr uint32_t PING_INTERVAL_MS = 20;   // 20ms -> 50Hz target CSI rate

// Only forward CSI from frames whose source MAC is the router.
// Downlink echo replies all come from the AP, so this drops noise frames.
constexpr bool    FILTER_ROUTER_MAC = true;
constexpr uint8_t ROUTER_MAC[6] = {0x2c, 0x30, 0x33, 0xd3, 0x61, 0xc3};

constexpr size_t  CSI_BUF_MAX = 512;   // covers 256/384-byte variants with headroom
constexpr size_t  CSI_QUEUE_LEN = 64;