#include "pinger.hpp"

#include "esp_log.h"
#include "ping/ping_sock.h"
#include "lwip/ip_addr.h"
#include "lwip/inet.h"

static const char* TAG = "ping";

bool Pinger::start(const char* target_ip, uint32_t interval_ms) {
    ip_addr_t target;
    if (!ipaddr_aton(target_ip, &target)) {
        ESP_LOGE(TAG, "bad target ip");
        return false;
    }

    esp_ping_config_t cfg = ESP_PING_DEFAULT_CONFIG();
    cfg.target_addr = target;
    cfg.count = 0;             // run forever
    cfg.interval_ms = interval_ms;
    cfg.timeout_ms = 200;
    cfg.task_stack_size = 3072;

    esp_ping_callbacks_t cbs = {};       // no per-ping callbacks needed

    esp_ping_handle_t handle;
    if (esp_ping_new_session(&cfg, &cbs, &handle) != ESP_OK) {
        ESP_LOGE(TAG, "new session failed");
        return false;
    }
    if (esp_ping_start(handle) != ESP_OK) {
        ESP_LOGE(TAG, "ping start failed");
        return false;
    }

    ESP_LOGI(TAG, "pinging %s every %lu ms",
             target_ip, static_cast<unsigned long>(interval_ms));
    return true;
}