#pragma once
#include <cstdint>

constexpr const char* WIFI_SSID = "NETGEAR82";
constexpr const char* WIFI_PASSWORD = "slowcanoe599";

constexpr const char* COLLECTOR_IP = "192.168.1.5";
constexpr uint16_t COLLECTOR_PORT = 4444; 

constexpr uint8_t SENSOR_ID = 1; 

constexpr uint8_t ROUTER_MAC[6] = {0x2c, 0x30, 0x33, 0xd3, 0x61, 0xc3};
constexpr size_t CSI_BUF_MAX = 256;

constexpr const char* ROUTER_IP = "192.168.1.1"; 
constexpr uint32_t   PING_INTERVAL_MS = 20;       