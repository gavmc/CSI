#include "udp_sender.hpp"

#include <cerrno>
#include "esp_log.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"

static const char* TAG ="udp";

UdpSender::~UdpSender(){
    if(sock_ >= 0) close(sock_);
}

bool UdpSender::start(const char* dest_ip, uint16_t dest_port){
    sock_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);

    if(sock_ < 0){
        ESP_LOGI(TAG, "socket() failed %d", errno);
        return false;
    }

    sockaddr_in dest = {};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(dest_port);
    dest.sin_addr.s_addr = inet_addr(dest_ip);

    if(connect(sock_, reinterpret_cast<sockaddr*>(&dest), sizeof(dest)) < 0){
        ESP_LOGI(TAG, "connect() failed, %d", errno);
        close(sock_);
        sock_ = -1;
        return false;
    }

    ESP_LOGI(TAG, "upd ready -> %s:%u", dest_ip, dest_port);
    return true;
}

bool UdpSender::send(const void* data, size_t len){
    if(sock_ < 0) return false;
    ssize_t n = ::send(sock_, data, len, 0);
    if(n < 0){
        ESP_LOGI(TAG, "send failed %d", errno);
        return false;
    }
    return true;
}