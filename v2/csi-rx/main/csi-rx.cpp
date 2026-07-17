#include <cstdio>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_netif.h"
#include "esp_event.h"

#include "config.hpp"
#include "wifi_manager.hpp"
#include "udp_sender.hpp"
#include "csi_capture.hpp"
#include "pinger.hpp"


static const char* TAG = "main";

static UdpSender sender;
static CsiCapture csi;
static Pinger pinger;

static void heartbeat_task(void*){
    uint32_t seq = 0;
    char buf[96];
    while(true){
        int len = std::snprintf(buf, sizeof(buf), 
        "heartbeat sensor=%u seq=%lu free_heap=%lu",
        SENSOR_ID,
        static_cast<unsigned long>(seq++),
        static_cast<unsigned long>(esp_get_free_heap_size())
    );
    sender.send(buf, len);
    vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

extern "C" void app_main(void)
{

    esp_err_t ret = nvs_flash_init();
    if(ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND){
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }

    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    static WifiManager wifi;
    wifi.start();
    wifi.wait_for_ip();

    if(!sender.start(COLLECTOR_IP, COLLECTOR_PORT)){
        ESP_LOGE(TAG, "udp sender failed to start");
        return;
    }

    xTaskCreate(heartbeat_task, "heartbeat", 4096, nullptr, 5, nullptr);

    if (!csi.start(&sender)) {
        ESP_LOGE(TAG, "csi failed to start");
        return;
    }

    pinger.start(ROUTER_IP, PING_INTERVAL_MS);

    ESP_LOGI(TAG, "system up. free heap: %lu bytes", 
        static_cast<unsigned long>(esp_get_free_heap_size())
    );

}
