echo 0 | sudo tee /proc/sys/net/ipv4/ip_forward

# Remove forwarding rules
sudo iptables -t nat -F

