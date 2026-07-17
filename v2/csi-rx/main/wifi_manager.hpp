#pragma once
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "esp_event.h"

class WifiManager{
    public:
    WifiManager();
    void start();
    void wait_for_ip();

    private:

    static void event_handler(void* arg, esp_event_base_t base, int32_t id, void* data);

    void handle_event(esp_event_base_t base, int32_t id, void* data);

    EventGroupHandle_t event_group_;
    int retry_count_ = 0;
    static constexpr int MAX_RETRIES = 10;
    static constexpr EventBits_t CONNECTED_BIT = BIT0;
};