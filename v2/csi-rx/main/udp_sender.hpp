#pragma once

#include <cstddef>
#include <cstdint>

class UdpSender{
    public:
    UdpSender() = default;
    ~UdpSender();
    UdpSender(const UdpSender&) = delete;
    UdpSender& operator = (const UdpSender&) = delete;

    bool start(const char* dest_ip, uint16_t dest_port);
    bool send(const void* data, size_t len);

    private:
    int sock_ = -1;
};