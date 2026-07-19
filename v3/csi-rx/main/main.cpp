#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "lwip/sockets.h"

#include "config.hpp"

static const char* TAG = "CSI";

#define WIFI_CONNECTED_BIT BIT0
static EventGroupHandle_t s_wifi_event_group;

// ---------------------------------------------------------------------------
// Wire format. Collector parses with:  struct.unpack('<HBBIqbBH', data[:20])
// ---------------------------------------------------------------------------
struct __attribute__((packed)) CsiHeader {
    uint16_t magic;        // 0xC51D
    uint8_t  sensor_id;    // SENSOR_ID from config
    uint8_t  channel;      // primary channel from rx_ctrl
    uint32_t seq;          // per-sensor sequence number (gap = dropped frame)
    int64_t  esp_time_us;  // esp_timer_get_time() at callback
    int8_t   rssi;
    uint8_t  flags;        // reserved
    uint16_t len;          // CSI payload bytes that follow
};
static_assert(sizeof(CsiHeader) == 20, "header size changed - update collector");

struct CsiItem {
    CsiHeader hdr;
    uint8_t   buf[CSI_BUF_MAX];
};

struct __attribute__((packed)) PingPacket {
    uint16_t magic;        // 0xB117
    uint8_t  sensor_id;
    uint8_t  pad;
    uint32_t seq;
    int64_t  esp_time_us;  // for clock-offset estimation on the collector
};

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
static QueueHandle_t s_csi_queue = nullptr;
static uint32_t s_seq = 0;
static uint32_t s_dropped_queue = 0;   // queue full
static uint32_t s_dropped_size = 0;    // payload too big / filtered handled separately
static uint32_t s_sent = 0;

// ---------------------------------------------------------------------------
// WiFi
// ---------------------------------------------------------------------------
static void event_handler(void*, esp_event_base_t base, int32_t id, void* data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGI(TAG, "Disconnected, retrying...");
        xEventGroupClearBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        auto* event = (ip_event_got_ip_t*) data;
        ESP_LOGI(TAG, "Connected! IP: " IPSTR, IP2STR(&event->ip_info.ip));
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

static void wifi_init()
{
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, event_handler, NULL, NULL);
    esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, event_handler, NULL, NULL);

    wifi_config_t wifi_config = {};
    strcpy((char*)wifi_config.sta.ssid,     WIFI_SSID);
    strcpy((char*)wifi_config.sta.password, WIFI_PASSWORD);

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT, false, true, portMAX_DELAY);
}

// ---------------------------------------------------------------------------
// CSI callback: runs in WiFi task context. Copy into queue and get out.
// No sockets, no logging, no blocking here.
// ---------------------------------------------------------------------------
static void csi_callback(void*, wifi_csi_info_t* csi)
{
    if (!csi || !csi->buf || csi->len == 0) return;

    if (FILTER_ROUTER_MAC && memcmp(csi->mac, ROUTER_MAC, 6) != 0) return;

    if ((size_t)csi->len > CSI_BUF_MAX) { s_dropped_size++; return; }

    CsiItem item;
    item.hdr.magic       = 0xC51D;
    item.hdr.sensor_id   = SENSOR_ID;
    item.hdr.channel     = csi->rx_ctrl.channel;
    item.hdr.seq         = s_seq++;
    item.hdr.esp_time_us = esp_timer_get_time();
    item.hdr.rssi        = csi->rx_ctrl.rssi;
    item.hdr.flags       = 0;
    item.hdr.len         = csi->len;
    memcpy(item.buf, csi->buf, csi->len);

    if (xQueueSend(s_csi_queue, &item, 0) != pdTRUE) {
        s_dropped_queue++;
    }
}

// ---------------------------------------------------------------------------
// Sender task: drains the queue, ships header+payload as one UDP datagram
// ---------------------------------------------------------------------------
static void sender_task(void*)
{
    struct sockaddr_in dest = {};
    dest.sin_family = AF_INET;
    dest.sin_port   = htons(CSI_PORT);
    inet_pton(AF_INET, COLLECTOR_IP, &dest.sin_addr);

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        ESP_LOGE(TAG, "sender socket() failed: errno %d", errno);
        vTaskDelete(NULL);
        return;
    }

    static CsiItem item;   // static: keep ~540B off this task's stack
    while (true) {
        if (xQueueReceive(s_csi_queue, &item, portMAX_DELAY) == pdTRUE) {
            size_t total = sizeof(CsiHeader) + item.hdr.len;
            int n = sendto(sock, &item, total, 0, (struct sockaddr*)&dest, sizeof(dest));
            if (n < 0) {
                if (errno == ENOMEM) vTaskDelay(pdMS_TO_TICKS(5));  // lwip buffers full, back off
            } else {
                s_sent++;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Ping task: generates downlink traffic (echo replies) to harvest CSI from,
// and carries local timestamps for clock-offset estimation on the collector
// ---------------------------------------------------------------------------
static void ping_task(void*)
{
    struct sockaddr_in dest = {};
    dest.sin_family = AF_INET;
    dest.sin_port   = htons(PING_PORT);
    inet_pton(AF_INET, COLLECTOR_IP, &dest.sin_addr);

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        ESP_LOGE(TAG, "ping socket() failed: errno %d", errno);
        vTaskDelete(NULL);
        return;
    }

    PingPacket pkt = {};
    pkt.magic     = 0xB117;
    pkt.sensor_id = SENSOR_ID;

    TickType_t last_wake = xTaskGetTickCount();
    while (true) {
        // Pause when disconnected instead of spamming a dead interface
        xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT, false, true, portMAX_DELAY);

        pkt.seq++;
        pkt.esp_time_us = esp_timer_get_time();
        sendto(sock, &pkt, sizeof(pkt), 0, (struct sockaddr*)&dest, sizeof(dest));

        vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(PING_INTERVAL_MS));
    }
}

// ---------------------------------------------------------------------------
// Stats task: one log line every 5s so you can watch rate/drops per sensor
// ---------------------------------------------------------------------------
static void stats_task(void*)
{
    uint32_t prev_sent = 0;
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(5000));
        uint32_t sent = s_sent;
        ESP_LOGI(TAG, "rate=%.1f Hz  sent=%lu  drop_q=%lu  drop_size=%lu  heap=%lu",
                 (sent - prev_sent) / 5.0f, sent, s_dropped_queue, s_dropped_size,
                 esp_get_free_heap_size());
        prev_sent = sent;
    }
}

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "CSI Sensor %d - ESP-IDF v6.0.1", SENSOR_ID);

    ESP_ERROR_CHECK(nvs_flash_init());
    wifi_init();

    esp_wifi_set_ps(WIFI_PS_NONE);

    // Everything that consumes CSI must exist BEFORE CSI is enabled
    s_csi_queue = xQueueCreate(CSI_QUEUE_LEN, sizeof(CsiItem));
    xTaskCreate(sender_task, "csi_send", 4096, NULL, 6, NULL);
    xTaskCreate(ping_task,   "ping",     4096, NULL, 5, NULL);
    xTaskCreate(stats_task,  "stats",    3072, NULL, 2, NULL);

    wifi_csi_config_t cfg = {};
    cfg.lltf_en           = true;
    cfg.htltf_en          = true;
    cfg.stbc_htltf2_en    = false;
    cfg.ltf_merge_en      = true;
    cfg.channel_filter_en = false;
    cfg.manu_scale        = false;

    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(csi_callback, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    ESP_LOGI(TAG, "CSI enabled, streaming to %s:%d", COLLECTOR_IP, CSI_PORT);
}