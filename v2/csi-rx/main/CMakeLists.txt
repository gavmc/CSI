idf_component_register(
    SRCS "csi_capture.cpp" "udp_sender.cpp" "csi-rx.cpp" "wifi_manager.cpp" "pinger.cpp"
    INCLUDE_DIRS "."
    REQUIRES esp_wifi nvs_flash esp_netif esp_event lwip esp_timer
)