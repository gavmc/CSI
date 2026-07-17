#include "wifi_manager.hpp"
#include "config.hpp"

#include <cstring>
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"

static const char* TAG = "wifi";

WifiManager::WifiManager(){
    event_group_ = xEventGroupCreate();
}

void WifiManager::start(){
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler, this, nullptr
    ));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &event_handler, this, nullptr
    ));

    wifi_config_t wifi_config = {};

    std::strncpy(reinterpret_cast<char*>(wifi_config.sta.ssid), WIFI_SSID, sizeof(wifi_config.sta.ssid));
    std::strncpy(reinterpret_cast<char*>(wifi_config.sta.password), WIFI_PASSWORD, sizeof(wifi_config.sta.password));

    wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "connecting to %s", WIFI_SSID);   
}


void WifiManager::wait_for_ip(){
    xEventGroupWaitBits(event_group_, CONNECTED_BIT, pdFALSE, pdTRUE, portMAX_DELAY);
}

void WifiManager::event_handler(void* arg, esp_event_base_t base, int32_t id, void* data){
    static_cast<WifiManager*>(arg)->handle_event(base, id, data);
}

void WifiManager::handle_event(esp_event_base_t base, int32_t id, void* data){
    if(base == WIFI_EVENT && id == WIFI_EVENT_STA_START){
        esp_wifi_connect();
    }else if(base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED){
        if(retry_count_ < MAX_RETRIES){
            esp_wifi_connect();
            retry_count_++;
            ESP_LOGI(TAG, "retry %d/%d", retry_count_, MAX_RETRIES);
        }else{
            ESP_LOGI(TAG, "failed after %d retries", MAX_RETRIES);
        }
    }else if(base == IP_EVENT && id == IP_EVENT_STA_GOT_IP){
        auto* event = static_cast<ip_event_got_ip_t*>(data);
        ESP_LOGI(TAG, "got ip: " IPSTR, IP2STR(&event->ip_info.ip));
        retry_count_ = 0;
        xEventGroupSetBits(event_group_, CONNECTED_BIT);
    }
}