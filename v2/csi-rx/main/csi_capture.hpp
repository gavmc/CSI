#pragma once
#include <cstdint>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "esp_wifi.h"

class UdpSender;

class CsiCapture{
    public:
    bool start(UdpSender* sender);
    private:
    static void csi_cb(void* ctx, wifi_csi_info_t* info);
    void on_csi(wifi_csi_info_t* info);

    static void sender_task(void* arg);
    void run_sender();

    UdpSender* sender_ = nullptr;
    QueueHandle_t queue_ = nullptr;

    uint32_t total_rx_ = 0;
    uint32_t total_match_ = 0;
    uint32_t total_dropq_ = 0;
};