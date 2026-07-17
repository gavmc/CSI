#include "csi_capture.hpp"

#include <cstring>
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"

#include "config.hpp"
#include "record.hpp"
#include "udp_sender.hpp"

static const char* TAG = "csi";

bool CsiCapture::start(UdpSender* sender){
    sender_ = sender;

    queue_ = xQueueCreate(32, sizeof(CsiRecord));
    if(!queue_){
        ESP_LOGI(TAG, "queue create failed");
        return false;
    }

    xTaskCreate(sender_task, "csi_send", 6144, this, 4, nullptr);

    wifi_promiscuous_filter_t prom_filter = {};
    prom_filter.filter_mask = WIFI_PROMIS_FILTER_MASK_MGMT | WIFI_PROMIS_FILTER_MASK_DATA;
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous_filter(&prom_filter));
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));

    wifi_csi_config_t cfg = {};
    cfg.lltf_en = true;
    cfg.htltf_en = true;
    cfg.stbc_htltf2_en = false;
    cfg.ltf_merge_en = false; // was true
    cfg.channel_filter_en = false; // was true
    cfg.manu_scale = false; 
    cfg.shift = 0;
    cfg.dump_ack_en = false;

    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(&csi_cb, this));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    ESP_LOGI(TAG, "csi enabled, filtering for "
            "%02x:%02x:%02x:%02x:%02x:%02x",
            ROUTER_MAC[0], ROUTER_MAC[1], ROUTER_MAC[2],
            ROUTER_MAC[3], ROUTER_MAC[4], ROUTER_MAC[5]
    );

    return true;
}


void CsiCapture::csi_cb(void* ctx, wifi_csi_info_t* info){
    static_cast<CsiCapture*>(ctx)->on_csi(info);
}

void CsiCapture::on_csi(wifi_csi_info_t* info){
    total_rx_++;

    if(std::memcmp(info->mac, ROUTER_MAC, 6) != 0){
        return;
    }

    total_match_++;

    CsiRecord rec = {};
    rec.magic = CSI_RECORD_MAGIC;
    rec.sensor_id = SENSOR_ID;
    rec.seq = total_match_;
    rec.timestamp_us = esp_timer_get_time();
    rec.rssi = info->rx_ctrl.rssi;
    rec.noise_floor = info->rx_ctrl.noise_floor;
    rec.channel = info->rx_ctrl.channel;
    rec.secondary_channel = info->rx_ctrl.secondary_channel;
    std::memcpy(rec.src_mac, info->mac, 6);
    rec.buf_len = info->len > CSI_BUF_MAX ? CSI_BUF_MAX : static_cast<uint16_t>(info->len);
    rec.first_word_invalid = info->first_word_invalid ? 1 : 0;
    std::memcpy(rec.buf, info->buf, rec.buf_len);

    if(xQueueSend(queue_, &rec, 0) != pdTRUE){
        total_dropq_++;
    }
}


void CsiCapture::sender_task(void* arg){
    static_cast<CsiCapture*>(arg)->run_sender();
}

void CsiCapture::run_sender(){
    constexpr size_t HEADER_SIZE = offsetof(CsiRecord, buf);

    CsiRecord rec;
    uint32_t last_log_ms = 0;

    while(true){
        if(xQueueReceive(queue_, &rec, pdMS_TO_TICKS(1000)) == pdTRUE){
            sender_->send(&rec, HEADER_SIZE + rec.buf_len);
        }

        uint32_t now = esp_log_timestamp();
        if(now - last_log_ms > 5000){
            ESP_LOGI(TAG, "rx=%lu match=%lu dropq=%lu queue=%u",
                    static_cast<unsigned long>(total_rx_),
                    static_cast<unsigned long>(total_match_),
                    static_cast<unsigned long>(total_dropq_),
                    static_cast<unsigned>(uxQueueMessagesWaiting(queue_))
            );
            last_log_ms = now;
        }
    }
}